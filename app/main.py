"""BechoAI FastAPI backend. Run: uvicorn app.main:app --reload   (docs at http://localhost:8000/docs)

Graded agent API (open, no login; format fixed by the hackathon):
    POST /chat     {"user_id", "thread_id", "message"} -> {"reply", "tool_calls", "tokens": {"in", "out"}}
    GET  /health   -> ok

Customer side (anonymous, token from POST /api/customer/session):
    GET  /api/stores/{store_id}                    store info
    GET  /api/stores/{store_id}/products           catalog (in stock), with filters
    POST /api/stores/{store_id}/chat               talk to the agent
    POST /api/stores/{store_id}/chat/confirm       confirm / cancel the order the agent prepared
    GET  /api/stores/{store_id}/chat/{chat_id}     chat history (reload the page)
    GET  /api/stores/{store_id}/my-orders          the customer's own orders
Seller side (JWT from signup / login; store_id always comes from the token):
    POST /api/auth/signup, /api/auth/login         GET /api/seller/me
    GET|POST /api/seller/products   PUT|DELETE /api/seller/products/{sku}   POST /api/seller/products/upload (CSV)
    GET /api/seller/orders   POST /api/seller/orders/{id}/approve | reject
    GET /api/seller/stats    GET /api/seller/conversations[/{thread_id}]    PUT /api/seller/settings
"""
from dotenv import load_dotenv

load_dotenv()  # before app.auth reads JWT_SECRET

import os  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pathlib import Path  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402

from app import chat, db, services  # noqa: E402
from app.auth import (current_customer_id, current_store_id, hash_password, new_customer_token,  # noqa: E402
                      seller_token, verify_password)
from app.db import Conversation, Seller, Store  # noqa: E402
from app.schemas import (AgentChatIn, AgentChatOut, ChatHistoryOut, ChatIn, ChatOut, ConfirmIn,  # noqa: E402
                         CustomerSessionOut, LoginIn,
                         ProductIn, SellerAuthOut, SettingsIn, SignupIn, StoreOut, UploadResult)
from app.services import ShopError  # noqa: E402

MAX_CSV_BYTES = 1_000_000
DEFAULT_STORE_ID = os.getenv("DEFAULT_STORE_ID", "noor-threads")    # the store POST /chat talks to
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.seed import seed
    seed()                                               # fresh database: create the demo store
    if getattr(app.state, "graph", None) is None:      # tests put a fake graph here first
        from app.agent import build_graph
        app.state.graph = build_graph()
    yield


app = FastAPI(title="BechoAI API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_methods=["*"], allow_headers=["*"],
                   allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()])
# No cookies are used (tokens go in the Authorization header), so "*" is safe for the public agent API.


@app.exception_handler(ShopError)
def shop_error(request: Request, exc: ShopError):
    return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_400_BAD_REQUEST)


def get_db():
    with db.session() as s:
        yield s


def get_graph(request: Request):
    return request.app.state.graph


def store_out(store: Store) -> StoreOut:
    return StoreOut(id=store.id, name=store.name, delivery_fees=store.delivery_fees,
                    default_delivery_fee=store.default_delivery_fee)


def existing_store(store_id: str, s=Depends(get_db)) -> Store:
    store = s.get(Store, store_id)
    if store is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Store not found.")
    return store


@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"


@app.get("/api/health")
def api_health():
    return {"ok": True}


def turn_to_chat_out(turn: chat.Turn) -> ChatOut:
    return ChatOut(reply=turn.reply, route=turn.route, products=turn.products, pending_order=turn.pending_order)


@app.post("/chat", response_model=AgentChatOut)
def agent_chat(body: AgentChatIn, graph=Depends(get_graph), s=Depends(get_db)):
    """The graded endpoint. user_id and thread_id come from the caller; both are escaped into the thread key,
    and long-term memory is keyed by (store, user_id), so users never see each other's chats or memories."""
    existing_store(DEFAULT_STORE_ID, s)
    turn = chat.run_turn(graph, DEFAULT_STORE_ID, body.user_id, body.thread_id, body.message)
    return AgentChatOut(reply=turn.reply, tool_calls=turn.tool_calls,
                        tokens={"in": turn.tokens_in, "out": turn.tokens_out})


# ── seller auth ──

@app.post("/api/auth/signup", response_model=SellerAuthOut, status_code=201)
def signup(body: SignupIn, s=Depends(get_db)):
    email = body.email.lower()
    if s.scalar(select(Seller).where(Seller.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists.")
    store = Store(id=services.make_store_id(s, body.store_name), name=body.store_name.strip(),
                  delivery_fees={}, default_delivery_fee=services.DEFAULT_DELIVERY_FEE)
    seller = Seller(email=email, password_hash=hash_password(body.password), store_id=store.id)
    try:
        s.add(store)
        s.flush()                                   # store row first: Postgres enforces the foreign key
        s.add(seller)
        s.commit()
    except IntegrityError:
        s.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Could not create the account, please try again.")
    return SellerAuthOut(token=seller_token(seller.id, store.id), store=store_out(store))


@app.post("/api/auth/login", response_model=SellerAuthOut)
def login(body: LoginIn, s=Depends(get_db)):
    seller = s.scalar(select(Seller).where(Seller.email == body.email.lower()))
    if seller is None or not verify_password(body.password, seller.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong email or password.")
    return SellerAuthOut(token=seller_token(seller.id, seller.store_id), store=store_out(s.get(Store, seller.store_id)))


@app.get("/api/seller/me", response_model=StoreOut)
def seller_me(store_id: str = Depends(current_store_id), s=Depends(get_db)):
    return store_out(existing_store(store_id, s))


# ── customer side ──

@app.post("/api/customer/session", response_model=CustomerSessionOut)
def customer_session():
    """Call once per browser and keep the token (e.g. localStorage). Send it as 'Authorization: Bearer ...'."""
    _, token = new_customer_token()
    return CustomerSessionOut(customer_token=token)


@app.get("/api/stores/{store_id}", response_model=StoreOut)
def get_store(store: Store = Depends(existing_store)):
    return store_out(store)


@app.get("/api/stores/{store_id}/products")
def store_products(store: Store = Depends(existing_store), query: str = "", category: str | None = None,
                   color: str | None = None, size: str | None = None, max_price: int | None = None,
                   s=Depends(get_db)):
    return services.search_catalog(s, store.id, query[:100], category, color, size, max_price, limit=200)


@app.post("/api/stores/{store_id}/chat", response_model=ChatOut)
def store_chat(body: ChatIn, store: Store = Depends(existing_store),
               customer_id: str = Depends(current_customer_id), graph=Depends(get_graph)):
    return turn_to_chat_out(chat.run_turn(graph, store.id, customer_id, body.chat_id, body.message))


@app.post("/api/stores/{store_id}/chat/confirm", response_model=ChatOut)
def store_chat_confirm(body: ConfirmIn, store: Store = Depends(existing_store),
                       customer_id: str = Depends(current_customer_id), graph=Depends(get_graph)):
    cfg = {"configurable": {"thread_id": chat.thread_id(store.id, customer_id, body.chat_id)}}
    if not graph.get_state(cfg).interrupts:
        raise HTTPException(status.HTTP_409_CONFLICT, "There is no order waiting for confirmation.")
    answer = "yes" if body.confirmed else "no"
    return turn_to_chat_out(chat.run_turn(graph, store.id, customer_id, body.chat_id, answer))


@app.get("/api/stores/{store_id}/chat/{chat_id}", response_model=ChatHistoryOut)
def store_chat_history(chat_id: str, store: Store = Depends(existing_store),
                       customer_id: str = Depends(current_customer_id), graph=Depends(get_graph)):
    return chat.history(graph, chat.thread_id(store.id, customer_id, chat_id))


@app.get("/api/stores/{store_id}/my-orders")
def my_orders(store: Store = Depends(existing_store), customer_id: str = Depends(current_customer_id),
              s=Depends(get_db)):
    return services.get_orders(s, store.id, customer_id)


# ── seller dashboard ──

@app.get("/api/seller/products")
def seller_products(store_id: str = Depends(current_store_id), s=Depends(get_db)):
    return services.list_products(s, store_id)


@app.post("/api/seller/products", status_code=201)
def create_product(body: ProductIn, store_id: str = Depends(current_store_id), s=Depends(get_db)):
    product, created = services.upsert_product(s, store_id, body.model_dump())
    if not created:
        raise HTTPException(status.HTTP_409_CONFLICT, f"{product['sku']} already exists; use PUT to change it.")
    return product


@app.put("/api/seller/products/{sku}")
def update_product(sku: str, body: ProductIn, store_id: str = Depends(current_store_id), s=Depends(get_db)):
    if body.sku.upper() != sku.upper():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "SKU in the URL and body must match.")
    services.check_stock(s, store_id, sku)                       # 400 if it isn't this store's product
    return services.upsert_product(s, store_id, body.model_dump())[0]


@app.delete("/api/seller/products/{sku}")
def remove_product(sku: str, store_id: str = Depends(current_store_id), s=Depends(get_db)):
    return {"sku": sku.upper(), "result": services.delete_product(s, store_id, sku)}


@app.post("/api/seller/products/upload", response_model=UploadResult)
async def upload_catalog(file: UploadFile = File(...), store_id: str = Depends(current_store_id), s=Depends(get_db)):
    raw = await file.read(MAX_CSV_BYTES + 1)
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "CSV must be under 1 MB.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "CSV must be UTF-8 text.")
    return services.import_csv(s, store_id, text)


@app.get("/api/seller/orders")
def list_orders(status_filter: str | None = Query(None, alias="status"), store_id: str = Depends(current_store_id),
                s=Depends(get_db)):
    return services.seller_orders(s, store_id, status_filter)


@app.post("/api/seller/orders/{order_id}/approve")
def approve_order(order_id: int, store_id: str = Depends(current_store_id), s=Depends(get_db)):
    return services.review_order(s, store_id, order_id, approve=True)


@app.post("/api/seller/orders/{order_id}/reject")
def reject_order(order_id: int, store_id: str = Depends(current_store_id), s=Depends(get_db)):
    return services.review_order(s, store_id, order_id, approve=False)


@app.get("/api/seller/stats")
def stats(store_id: str = Depends(current_store_id), s=Depends(get_db)):
    return services.store_stats(s, store_id)


@app.get("/api/seller/conversations")
def conversations(store_id: str = Depends(current_store_id), s=Depends(get_db)):
    rows = s.scalars(select(Conversation).where(Conversation.store_id == store_id)
                     .order_by(Conversation.updated_at.desc()).limit(200))
    return [{"thread_id": c.thread_id, "customer": c.user_id[-6:], "last_message": c.last_message,
             "turns": c.turns, "updated_at": c.updated_at.strftime("%Y-%m-%d %H:%M")} for c in rows]


@app.get("/api/seller/conversations/{thread_id}", response_model=ChatHistoryOut)
def conversation(thread_id: str, store_id: str = Depends(current_store_id), s=Depends(get_db),
                 graph=Depends(get_graph)):
    conv = s.get(Conversation, thread_id)
    if conv is None or conv.store_id != store_id:               # another store's chat looks like a missing one
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")
    return chat.history(graph, thread_id)


@app.put("/api/seller/settings", response_model=StoreOut)
def update_settings(body: SettingsIn, store_id: str = Depends(current_store_id), s=Depends(get_db)):
    return store_out(services.update_settings(s, store_id, body.name, body.delivery_fees, body.default_delivery_fee))


# ── website (also deployable separately on Vercel from the web/ folder) ──

if WEB_DIR.exists():
    @app.get("/s/{store_id}", include_in_schema=False)
    def store_page(store_id: str):
        return FileResponse(WEB_DIR / "store.html")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard_page():
        return FileResponse(WEB_DIR / "dashboard.html")

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")   # last: index.html, css, js
