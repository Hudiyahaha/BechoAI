"""Request and response shapes for the API (validated by FastAPI / Pydantic)."""
from pydantic import BaseModel, EmailStr, Field

SKU = Field(pattern=r"^[A-Za-z0-9-]{1,32}$")
CHAT_ID = Field(pattern=r"^[A-Za-z0-9-]{1,64}$", description="Chosen by the browser, one per chat window")
MAX_MESSAGE_CHARS = 1000


# ── auth ──

class SignupIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    store_name: str = Field(min_length=2, max_length=80)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class StoreOut(BaseModel):
    id: str
    name: str
    delivery_fees: dict[str, int]
    default_delivery_fee: int


class SellerAuthOut(BaseModel):
    token: str
    store: StoreOut


class CustomerSessionOut(BaseModel):
    customer_token: str


# ── catalog ──

class ProductIn(BaseModel):
    sku: str = SKU
    name: str = Field(min_length=2, max_length=120)
    category: str = Field(min_length=2, max_length=40)
    color: str = Field(min_length=2, max_length=40)
    price: int = Field(gt=0, le=10_000_000, description="PKR")
    stock: dict[str, int] = Field(description='{"S": 2, "M": 5} or {"ONE SIZE": 8}')
    description: str = Field(default="", max_length=500)
    image_url: str | None = Field(default=None, max_length=500)


class ProductOut(ProductIn):
    sizes_in_stock: dict[str, int] | None = None


class UploadResult(BaseModel):
    created: int
    updated: int
    errors: list[str]


class SettingsIn(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    delivery_fees: dict[str, int] = Field(description='{"islamabad": 200}')
    default_delivery_fee: int = Field(ge=0, le=100_000)


# ── chat ──

class ChatIn(BaseModel):
    chat_id: str = CHAT_ID
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ConfirmIn(BaseModel):
    chat_id: str = CHAT_ID
    confirmed: bool


class PendingOrder(BaseModel):
    summary: str
    total: int
    customer_name: str
    phone: str
    address: str
    city: str


class ChatOut(BaseModel):
    reply: str
    route: str | None = None
    products: list[dict] = []                 # product cards from search_catalog in this turn
    pending_order: PendingOrder | None = None  # set when the customer must confirm in the UI


class ChatMessage(BaseModel):
    role: str                                 # "customer" | "assistant"
    text: str


class ChatHistoryOut(BaseModel):
    messages: list[ChatMessage]
    pending_order: PendingOrder | None = None
