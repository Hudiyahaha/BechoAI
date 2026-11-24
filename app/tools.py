"""The agent's tools. Each one reads store_id / user_id from runtime.context, never from a model argument."""
from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app import db, services
from app.context import ShopContext
from app.services import ShopError


class CartItem(BaseModel):
    sku: str = Field(description="Product code from search_catalog, e.g. 'NT-101'")
    size: str = Field(description="Size such as 'S', 'M', 'L', 'XL', or 'ONE SIZE' for unstitched/dupattas")
    qty: int = Field(default=1, ge=1, le=services.MAX_QTY_PER_ITEM)


def _fmt_product(p: dict) -> str:
    if list(p["sizes_in_stock"]) == [services.ONE_SIZE]:
        sizes = f"one size, fits every size: {p['sizes_in_stock'][services.ONE_SIZE]}"
    else:
        sizes = ", ".join(f"{k}: {v}" for k, v in p["sizes_in_stock"].items())
    return (f"{p['sku']} | {p['name']} | {p['color']} {p['category']} | Rs {p['price']:,} | sizes {sizes}"
            f" | {p['description']}")


def _fmt_quote(q: dict) -> str:
    lines = [f"{l['qty']} x {l['name']} ({l['sku']}, {l['size']}) @ Rs {l['unit_price']:,} = Rs {l['line_total']:,}"
             for l in q["lines"]]
    return "\n".join(lines + [f"Subtotal: Rs {q['subtotal']:,}",
                              f"Delivery to {q['city']}: Rs {q['delivery_fee']:,}",
                              f"Total (cash on delivery): Rs {q['total']:,}"])


def _fmt_order(o: dict) -> str:
    items = ", ".join(f"{i['qty']} x {i['name']} ({i['size']})" for i in o["items"])
    return f"Order #{o['order_id']} | {o['status']} | {o['created_at']} | Rs {o['total']:,} | {items}"


@tool(response_format="content_and_artifact")
def search_catalog(runtime: ToolRuntime[ShopContext], query: str = "", category: str | None = None,
                   color: str | None = None, size: str | None = None, max_price: int | None = None):
    """Search this store's in-stock products. Put colour, size, category and budget (max_price in PKR)
    in their own arguments; use query for other words like 'eid' or 'embroidered'."""
    store_id, note = runtime.context.store_id, ""
    with db.session() as s:
        products = services.search_catalog(s, store_id, query, category, color, size, max_price)
        if not products and query:   # words like "new arrival" match nothing: keep the filters, drop the words
            products = services.search_catalog(s, store_id, "", category, color, size, max_price)
            note = f"No product mentions '{query}'. Other in-stock items with the same filters:\n" if products else ""
    if not products:
        return "No matching products in stock. Try fewer filters.", []
    return note + "\n".join(_fmt_product(p) for p in products), products


@tool
def check_stock(sku: str, runtime: ToolRuntime[ShopContext], size: str | None = None) -> str:
    """Check how many pieces of a product (by code, e.g. 'NT-101') are left, optionally for one size."""
    try:
        with db.session() as s:
            r = services.check_stock(s, runtime.context.store_id, sku, size)
    except ShopError as e:
        return str(e)
    if "available" in r:
        return f"{r['name']} ({r['sku']}) size {r['size']}: {r['available']} available."
    sizes = ", ".join(f"{k}: {v}" for k, v in r["sizes_in_stock"].items()) or "out of stock"
    return f"{r['name']} ({r['sku']}): {sizes}"


@tool
def calculate_total(items: list[CartItem], city: str, runtime: ToolRuntime[ShopContext]) -> str:
    """Price the cart with delivery to the customer's city. Always show this to the customer before ordering."""
    try:
        with db.session() as s:
            quote = services.calculate_total(s, runtime.context.store_id, [i.model_dump() for i in items], city)
    except ShopError as e:
        return str(e)
    return _fmt_quote(quote)


@tool
def place_order(items: list[CartItem], customer_name: str, phone: str, address: str, city: str,
                runtime: ToolRuntime[ShopContext]) -> str:
    """Place a cash-on-delivery order. Use only after the customer has seen calculate_total and given name,
    phone, address and city. The customer is asked to confirm, then the seller must approve the order."""
    ctx = runtime.context
    cart = [i.model_dump() for i in items]
    try:
        with db.session() as s:
            services.validate_customer(customer_name, phone, address, city)
            quote = services.calculate_total(s, ctx.store_id, cart, city)
    except ShopError as e:
        return str(e)

    # Human-in-the-loop #1: the customer confirms in the UI. Nothing is saved before this.
    args = {"items": cart, "customer_name": customer_name, "phone": phone, "address": address, "city": city}
    answer = interrupt({"type": "confirm_order", "summary": _fmt_quote(quote), "total": quote["total"],
                        "customer_name": customer_name, "phone": phone, "address": address, "city": city,
                        "args": args})
    if not (isinstance(answer, dict) and answer.get("confirmed") is True):
        return "The customer did not confirm. No order was placed."

    try:
        with db.session() as s:
            order = services.place_order(s, ctx.store_id, ctx.user_id, cart, customer_name, phone, address, city)
    except ShopError as e:                                        # e.g. stock sold out while confirming
        return str(e)
    # Human-in-the-loop #2: the seller approves or rejects it from the dashboard.
    return (f"Order #{order['order_id']} received (total Rs {order['total']:,}). "
            "It is waiting for the seller's approval; the customer can ask for its status any time.")


@tool
def get_order_status(runtime: ToolRuntime[ShopContext], order_id: int | None = None) -> str:
    """Status of the customer's own order by number, or their latest orders if no number is given."""
    try:
        with db.session() as s:
            orders = services.get_orders(s, runtime.context.store_id, runtime.context.user_id, order_id)
    except ShopError as e:
        return str(e)
    return "\n".join(_fmt_order(o) for o in orders) or "You have no orders with this store yet."


TOOLS = [search_catalog, check_stock, calculate_total, place_order, get_order_status]
