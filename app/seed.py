"""Demo data: one fake clothing store. Run: python -m app.seed  (all data is made up).
Set DEMO_SELLER_EMAIL and DEMO_SELLER_PASSWORD in .env to also create a dashboard login for it."""
import os

from dotenv import load_dotenv
from sqlalchemy import select

from app import db
from app.auth import hash_password
from app.db import Product, Seller, Store

DEMO_STORE = {"id": "noor-threads", "name": "Noor Threads (demo)",
              "delivery_fees": {"islamabad": 200, "rawalpindi": 200}, "default_delivery_fee": 350}

S = lambda s, m, l, xl: {"S": s, "M": m, "L": l, "XL": xl}
ONE = lambda n: {"ONE SIZE": n}

DEMO_PRODUCTS = [
    # sku, name, category, color, price, stock, description
    ("NT-101", "Gulnaar Lawn 3-Piece", "unstitched", "red", 4500, ONE(12), "Printed lawn shirt, chiffon dupatta, cambric trouser. Eid collection."),
    ("NT-102", "Sehar Lawn 3-Piece", "unstitched", "pastel pink", 3900, ONE(8), "Soft pastel print with embroidered neckline."),
    ("NT-103", "Neelam Lawn 2-Piece", "unstitched", "blue", 2900, ONE(0), "Shirt and dupatta, digital print."),
    ("NT-201", "Rubab Embroidered Kurta", "kurta", "red", 3200, S(2, 4, 3, 1), "Cotton kurta with thread embroidery. Eid collection."),
    ("NT-202", "Zoya Straight Kurta", "kurta", "white", 2500, S(5, 6, 4, 2), "Everyday cotton kurta, chikankari style."),
    ("NT-203", "Mehr Block-Print Kurta", "kurta", "mustard", 2200, S(0, 3, 2, 0), "Hand block-printed cotton."),
    ("NT-204", "Hira Linen Kurta", "kurta", "black", 3600, S(1, 2, 2, 1), "Winter linen kurta with lace detail."),
    ("NT-301", "Shehnaz Formal Suit", "formal", "maroon", 9500, S(1, 2, 1, 0), "Chiffon formal with heavy embroidery, for weddings."),
    ("NT-302", "Anaya Velvet Formal", "formal", "emerald green", 12500, S(0, 1, 1, 1), "Velvet shirt with embroidered shawl."),
    ("NT-303", "Laila Organza Formal", "formal", "peach", 8800, S(2, 2, 2, 1), "Organza shirt, raw silk trouser."),
    ("NT-401", "Chiffon Dupatta", "dupatta", "red", 1200, ONE(15), "Plain chiffon dupatta with lace border. Matches NT-101 and NT-201."),
    ("NT-402", "Banarsi Dupatta", "dupatta", "gold", 2400, ONE(6), "Banarsi weave, great with formals."),
    ("NT-403", "Printed Lawn Dupatta", "dupatta", "white", 900, ONE(10), "Lightweight printed lawn dupatta."),
    ("NT-501", "Cigarette Pants", "trouser", "white", 1500, S(4, 5, 5, 3), "Stretch cotton cigarette pants."),
    ("NT-502", "Tulip Shalwar", "trouser", "black", 1400, S(3, 4, 4, 2), "Cotton tulip shalwar."),
]


def seed(store: dict = DEMO_STORE, products: list = DEMO_PRODUCTS):
    with db.session() as s:
        if s.get(Store, store["id"]):
            return False
        s.add(Store(**store))
        for sku, name, category, color, price, stock, desc in products:
            s.add(Product(store_id=store["id"], sku=sku, name=name, category=category, color=color,
                          price=price, stock=stock, description=desc))
        s.commit()
        return True


def seed_demo_seller(email: str | None, password: str | None, store_id: str = DEMO_STORE["id"]) -> str:
    if not email or not password:
        return "No demo seller login created (set DEMO_SELLER_EMAIL and DEMO_SELLER_PASSWORD in .env)."
    with db.session() as s:
        if s.scalar(select(Seller).where(Seller.email == email.lower())):
            return "Demo seller login already exists."
        s.add(Seller(email=email.lower(), password_hash=hash_password(password), store_id=store_id))
        s.commit()
    return f"Created demo seller login {email.lower()}."


if __name__ == "__main__":
    load_dotenv()
    print("Seeded demo store." if seed() else "Demo store already exists.")
    print(seed_demo_seller(os.getenv("DEMO_SELLER_EMAIL"), os.getenv("DEMO_SELLER_PASSWORD")))
