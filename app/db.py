"""Database models and session. DATABASE_URL picks the DB: SQLite locally, Postgres when deployed."""
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import JSON, DateTime, ForeignKey, String, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = f"sqlite:///{ROOT / 'bechoai.db'}"

engine = None
SessionLocal = None


class Base(DeclarativeBase):
    pass


class Store(Base):
    """One seller = one store. Every other table points at a store_id."""
    __tablename__ = "stores"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)          # slug, e.g. "noor-threads"
    name: Mapped[str]
    delivery_fees: Mapped[dict] = mapped_column(JSON, default=dict)        # {"islamabad": 200, ...}
    default_delivery_fee: Mapped[int] = mapped_column(default=350)         # PKR, any other city


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (UniqueConstraint("store_id", "sku"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[str] = mapped_column(ForeignKey("stores.id"), index=True)
    sku: Mapped[str] = mapped_column(String(32))                           # code customers see, e.g. "NT-101"
    name: Mapped[str]
    category: Mapped[str]
    color: Mapped[str]
    description: Mapped[str] = mapped_column(default="")
    price: Mapped[int]                                                     # PKR
    stock: Mapped[dict] = mapped_column(JSON)                              # {"S": 2, "M": 5} or {"ONE SIZE": 8}
    image_url: Mapped[str | None] = mapped_column(default=None)


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[str] = mapped_column(ForeignKey("stores.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(200), index=True)
    customer_name: Mapped[str]
    phone: Mapped[str]
    address: Mapped[str]
    city: Mapped[str]
    status: Mapped[str] = mapped_column(default="awaiting_approval")
    subtotal: Mapped[int]
    delivery_fee: Mapped[int]
    total: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    items: Mapped[list["OrderItem"]] = relationship(back_populates="order", cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "order_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    sku: Mapped[str]
    name: Mapped[str]
    size: Mapped[str]
    qty: Mapped[int]
    unit_price: Mapped[int]                                                # price at the time of ordering
    order: Mapped[Order] = relationship(back_populates="items")


class Seller(Base):
    """A seller account. One seller owns one store; the dashboard token carries this store_id."""
    __tablename__ = "sellers"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str]
    store_id: Mapped[str] = mapped_column(ForeignKey("stores.id"), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Conversation(Base):
    """Index of chats for the seller dashboard. The messages themselves live in the LangGraph checkpointer."""
    __tablename__ = "conversations"
    thread_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    store_id: Mapped[str] = mapped_column(ForeignKey("stores.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(200), index=True)
    last_message: Mapped[str] = mapped_column(default="")
    turns: Mapped[int] = mapped_column(default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


def postgres_url() -> str | None:
    """The Postgres URL (Supabase pooler) if one is configured, in plain psycopg form."""
    url = os.getenv("DATABASE_URL") or ""
    if url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
        return "postgresql://" + url.split("://", 1)[1]
    return None


def configure(url: str | None = None):
    """(Re)connect to a database and create the tables. Tests call this with a temp DB."""
    global engine, SessionLocal
    url = url or os.getenv("DATABASE_URL") or DEFAULT_URL
    if url.startswith(("postgres://", "postgresql://")):
        # SQLAlchemy needs the psycopg driver named; poolers (Supabase/pgbouncer) don't allow prepared statements.
        engine = create_engine("postgresql+psycopg://" + url.split("://", 1)[1], pool_pre_ping=True,
                               pool_size=5, max_overflow=5, connect_args={"prepare_threshold": None})
    else:
        engine = create_engine(url)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    return engine


def session():
    if SessionLocal is None:
        configure()
    return SessionLocal()
