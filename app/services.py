"""Business logic behind the agent's tools and the seller dashboard.

store_id and user_id are always passed in by the server (from login / runtime context), never by the LLM.
Prices and totals are always read from the database, never taken from the model.
"""
import re

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.db import Order, OrderItem, Product, Store

ONE_SIZE = "ONE SIZE"
MAX_QTY_PER_ITEM = 10
DEFAULT_DELIVERY_FEE = 250        # PKR for new stores until the seller sets their own
PHONE_RE = re.compile(r"^(03\d{9}|\+923\d{9})$")            # 03001234567 or +923001234567


class ShopError(ValueError):
    """A problem the customer can fix (bad SKU, out of stock, wrong phone). The message is safe to show."""


# ── helpers ──

def _in_stock_sizes(p: Product) -> dict:
    return {size: qty for size, qty in p.stock.items() if qty > 0}


def product_to_dict(p: Product) -> dict:
    return {"sku": p.sku, "name": p.name, "category": p.category, "color": p.color, "price": p.price,
            "sizes_in_stock": _in_stock_sizes(p), "description": p.description, "image_url": p.image_url}


def _get_store(s: Session, store_id: str) -> Store:
    store = s.get(Store, store_id)
    if store is None:
        raise ShopError("Store not found.")
    return store


def _get_product(s: Session, store_id: str, sku: str) -> Product:
    p = s.scalar(select(Product).where(Product.store_id == store_id, Product.sku == sku.strip().upper()))
    if p is None:
        raise ShopError(f"No product with code {sku} in this store.")
    return p


def _norm_size(size: str) -> str:
    return size.strip().upper()


def order_to_dict(o: Order) -> dict:
    return {"order_id": o.id, "status": o.status, "customer_name": o.customer_name, "city": o.city,
            "subtotal": o.subtotal, "delivery_fee": o.delivery_fee, "total": o.total,
            "created_at": o.created_at.strftime("%Y-%m-%d %H:%M"),
            "items": [{"sku": i.sku, "name": i.name, "size": i.size, "qty": i.qty, "unit_price": i.unit_price}
                      for i in o.items]}


# ── tool logic ──

def search_catalog(s: Session, store_id: str, query: str = "", category: str | None = None,
                   color: str | None = None, size: str | None = None, max_price: int | None = None,
                   limit: int = 8) -> list[dict]:
    """In-stock products of ONE store matching the filters, cheapest first."""
    q = select(Product).where(Product.store_id == store_id)
    if category:
        q = q.where(Product.category.ilike(f"%{category}%"))
    if color:
        q = q.where(Product.color.ilike(f"%{color}%"))
    if max_price is not None:
        q = q.where(Product.price <= max_price)
    words = query.lower().split()
    if words:   # match ANY word ("dupatta trouser" finds both); best matches first
        q = q.where(or_(*[f.ilike(f"%{w}%") for w in words for f in
                          (Product.name, Product.description, Product.category, Product.color, Product.sku)]))

    def score(p: Product) -> int:
        text = f"{p.name} {p.description} {p.category} {p.color} {p.sku}".lower()
        return sum(w in text for w in words)

    results = []
    for p in sorted(s.scalars(q), key=lambda p: (-score(p), p.price)):
        sizes = _in_stock_sizes(p)
        if not sizes:
            continue
        if size and _norm_size(size) not in sizes and list(sizes) != [ONE_SIZE]:
            continue
        results.append(product_to_dict(p))
        if len(results) >= limit:
            break
    return results


def check_stock(s: Session, store_id: str, sku: str, size: str | None = None) -> dict:
    p = _get_product(s, store_id, sku)
    sizes = _in_stock_sizes(p)
    if size is None:
        return {"sku": p.sku, "name": p.name, "sizes_in_stock": sizes}
    key = ONE_SIZE if list(p.stock) == [ONE_SIZE] else _norm_size(size)
    return {"sku": p.sku, "name": p.name, "size": key, "available": p.stock.get(key, 0)}


def calculate_total(s: Session, store_id: str, items: list[dict], city: str) -> dict:
    """Price a cart from the DB. items: [{"sku", "size", "qty"}]. Raises ShopError if anything is invalid."""
    if not items:
        raise ShopError("The cart is empty.")
    store = _get_store(s, store_id)
    lines, subtotal = [], 0
    for item in items:
        qty = int(item.get("qty", 1))
        if not 1 <= qty <= MAX_QTY_PER_ITEM:
            raise ShopError(f"Quantity must be between 1 and {MAX_QTY_PER_ITEM}.")
        p = _get_product(s, store_id, item["sku"])
        size = ONE_SIZE if list(p.stock) == [ONE_SIZE] else _norm_size(item.get("size", ""))
        available = p.stock.get(size, 0)
        if available < qty:
            raise ShopError(f"{p.name} ({p.sku}) size {size}: only {available} left.")
        line_total = p.price * qty
        subtotal += line_total
        lines.append({"sku": p.sku, "name": p.name, "size": size, "qty": qty,
                      "unit_price": p.price, "line_total": line_total, "product_id": p.id})
    delivery_fee = store.delivery_fees.get(city.strip().lower(), store.default_delivery_fee)
    return {"lines": lines, "subtotal": subtotal, "city": city.strip().title(),
            "delivery_fee": delivery_fee, "total": subtotal + delivery_fee}


def validate_customer(customer_name: str, phone: str, address: str, city: str) -> str:
    """Returns the cleaned phone number, or raises ShopError."""
    phone = re.sub(r"[\s-]", "", phone)
    if not PHONE_RE.match(phone):
        raise ShopError("Please give a valid Pakistani mobile number, e.g. 03001234567.")
    if len(customer_name.strip()) < 2 or len(address.strip()) < 8 or not city.strip():
        raise ShopError("Please give your full name, complete address and city.")
    return phone


def place_order(s: Session, store_id: str, user_id: str, items: list[dict], customer_name: str,
                phone: str, address: str, city: str) -> dict:
    """Create the order as 'awaiting_approval' and reserve the stock. The seller approves it later."""
    phone = validate_customer(customer_name, phone, address, city)
    quote = calculate_total(s, store_id, items, city)            # re-priced from the DB, never trusted from input
    order = Order(store_id=store_id, user_id=user_id, customer_name=customer_name.strip(), phone=phone,
                  address=address.strip(), city=quote["city"], subtotal=quote["subtotal"],
                  delivery_fee=quote["delivery_fee"], total=quote["total"])
    for line in quote["lines"]:
        p = s.get(Product, line["product_id"])
        p.stock = {**p.stock, line["size"]: p.stock[line["size"]] - line["qty"]}   # reserve stock
        order.items.append(OrderItem(product_id=p.id, sku=p.sku, name=p.name, size=line["size"],
                                     qty=line["qty"], unit_price=line["unit_price"]))
    s.add(order)
    s.commit()
    return order_to_dict(order)


def get_orders(s: Session, store_id: str, user_id: str, order_id: int | None = None) -> list[dict]:
    """A customer's own orders in this store. Another customer's order looks exactly like a missing one."""
    q = select(Order).where(Order.store_id == store_id, Order.user_id == user_id)
    if order_id is not None:
        q = q.where(Order.id == order_id)
    orders = s.scalars(q.order_by(Order.created_at.desc()).limit(5)).all()
    if order_id is not None and not orders:
        raise ShopError(f"I couldn't find order #{order_id} for you.")
    return [order_to_dict(o) for o in orders]


# ── seller dashboard ──

def review_order(s: Session, store_id: str, order_id: int, approve: bool) -> dict:
    """Seller approves or rejects a pending order. Rejecting puts the reserved stock back."""
    order = s.scalar(select(Order).where(Order.store_id == store_id, Order.id == order_id))
    if order is None:
        raise ShopError(f"Order #{order_id} not found.")
    if order.status != "awaiting_approval":
        raise ShopError(f"Order #{order_id} is already {order.status}.")
    order.status = "approved" if approve else "rejected"
    if not approve:
        for item in order.items:
            p = s.get(Product, item.product_id)
            p.stock = {**p.stock, item.size: p.stock.get(item.size, 0) + item.qty}
    s.commit()
    return order_to_dict(order)


# ── seller: store, catalog, orders, stats ──

def make_store_id(s: Session, name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "store"
    slug, n = base, 1
    while s.get(Store, slug) is not None:
        n += 1
        slug = f"{base}-{n}"
    return slug


def update_settings(s: Session, store_id: str, name: str, delivery_fees: dict, default_delivery_fee: int) -> Store:
    store = _get_store(s, store_id)
    fees = {city.strip().lower(): int(fee) for city, fee in delivery_fees.items() if city.strip()}
    if any(fee < 0 for fee in fees.values()):
        raise ShopError("Delivery fees can't be negative.")
    store.name, store.delivery_fees, store.default_delivery_fee = name.strip(), fees, default_delivery_fee
    s.commit()
    return store


def _clean_stock(stock: dict) -> dict:
    clean = {}
    for size, qty in stock.items():
        size, qty = _norm_size(str(size)), int(qty)
        if not size or qty < 0:
            raise ShopError("Stock needs a size and a quantity of 0 or more.")
        clean[size] = qty
    if not clean:
        raise ShopError("Give stock for at least one size (or ONE SIZE).")
    if ONE_SIZE in clean and len(clean) > 1:
        raise ShopError("Use either ONE SIZE or separate sizes, not both.")
    return clean


def list_products(s: Session, store_id: str) -> list[dict]:
    return [{**product_to_dict(p), "stock": p.stock}
            for p in s.scalars(select(Product).where(Product.store_id == store_id).order_by(Product.sku))]


def upsert_product(s: Session, store_id: str, data: dict) -> tuple[dict, bool]:
    """Create or update by SKU. Returns (product, created)."""
    sku = data["sku"].strip().upper()
    p = s.scalar(select(Product).where(Product.store_id == store_id, Product.sku == sku))
    created = p is None
    if created:
        p = Product(store_id=store_id, sku=sku)
        s.add(p)
    p.name, p.category, p.color = data["name"].strip(), data["category"].strip().lower(), data["color"].strip().lower()
    p.price, p.stock = int(data["price"]), _clean_stock(data["stock"])
    p.description, p.image_url = (data.get("description") or "").strip(), data.get("image_url") or None
    if p.price <= 0:
        raise ShopError("Price must be more than 0.")
    s.commit()
    return {**product_to_dict(p), "stock": p.stock}, created


def delete_product(s: Session, store_id: str, sku: str) -> str:
    """Delete a product. One that was ever ordered is kept for the order history, with its stock set to 0."""
    p = _get_product(s, store_id, sku)
    if s.scalar(select(OrderItem.id).where(OrderItem.product_id == p.id).limit(1)) is not None:
        p.stock = {size: 0 for size in p.stock}
        s.commit()
        return "archived"
    s.delete(p)
    s.commit()
    return "deleted"


CSV_COLUMNS = ["sku", "name", "category", "color", "price", "stock", "description", "image_url"]
MAX_CSV_ROWS = 1000


def parse_stock(text: str) -> dict:
    """'S:2;M:4;L:0' -> sizes, or a plain number '8' -> ONE SIZE."""
    text = text.strip()
    if text.isdigit():
        return {ONE_SIZE: int(text)}
    try:
        return {size.strip(): int(qty) for size, qty in (part.split(":") for part in text.split(";") if part.strip())}
    except ValueError:
        raise ShopError(f"Stock '{text}' should look like S:2;M:4 or a single number.")


def import_csv(s: Session, store_id: str, text: str) -> dict:
    import csv
    import io
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in CSV_COLUMNS[:6] if c not in (reader.fieldnames or [])]
    if missing:
        raise ShopError(f"CSV is missing columns: {', '.join(missing)}. Expected: {', '.join(CSV_COLUMNS)}.")
    created = updated = 0
    errors = []
    for line, row in enumerate(reader, start=2):
        if line - 1 > MAX_CSV_ROWS:
            errors.append(f"Stopped after {MAX_CSV_ROWS} rows.")
            break
        try:
            if not re.fullmatch(r"[A-Za-z0-9-]{1,32}", (row.get("sku") or "").strip()):
                raise ShopError("SKU must be 1-32 letters, digits or dashes.")
            if len((row.get("name") or "").strip()) < 2:
                raise ShopError("Name is missing.")
            data = {**row, "price": int(float(row["price"])), "stock": parse_stock(row["stock"] or "")}
            _, was_created = upsert_product(s, store_id, data)
            created, updated = created + was_created, updated + (not was_created)
        except (ShopError, ValueError, KeyError, TypeError) as e:
            s.rollback()
            errors.append(f"Row {line}: {e}")
    return {"created": created, "updated": updated, "errors": errors}


def seller_orders(s: Session, store_id: str, status: str | None = None, limit: int = 200) -> list[dict]:
    """The seller sees the delivery details (phone, address) that customers' chat tools never show."""
    q = select(Order).where(Order.store_id == store_id)
    if status:
        q = q.where(Order.status == status)
    return [{**order_to_dict(o), "phone": o.phone, "address": o.address}
            for o in s.scalars(q.order_by(Order.created_at.desc()).limit(limit))]


def store_stats(s: Session, store_id: str) -> dict:
    from app.db import Conversation
    orders = s.scalars(select(Order).where(Order.store_id == store_id)).all()
    by_status: dict[str, int] = {}
    for o in orders:
        by_status[o.status] = by_status.get(o.status, 0) + 1
    approved = [o for o in orders if o.status == "approved"]
    sold: dict[str, dict] = {}
    for o in approved:
        for i in o.items:
            row = sold.setdefault(i.sku, {"sku": i.sku, "name": i.name, "qty": 0, "revenue": 0})
            row["qty"] += i.qty
            row["revenue"] += i.qty * i.unit_price
    chats = len(s.scalars(select(Conversation.thread_id).where(Conversation.store_id == store_id)).all())
    low_stock = [{"sku": p.sku, "name": p.name, "left": sum(p.stock.values())}
                 for p in s.scalars(select(Product).where(Product.store_id == store_id))
                 if 0 < sum(p.stock.values()) <= 2]
    return {"orders_by_status": by_status, "revenue": sum(o.total for o in approved),
            "pending_value": sum(o.total for o in orders if o.status == "awaiting_approval"),
            "top_products": sorted(sold.values(), key=lambda r: -r["qty"])[:5],
            "conversations": chats,
            "chat_to_order_rate": round(len(orders) / chats, 2) if chats else 0.0,
            "low_stock": low_stock}
