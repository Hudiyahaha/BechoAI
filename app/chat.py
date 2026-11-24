"""Runs the LangGraph agent for one HTTP request and turns its state into API responses."""
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app import db
from app.context import ShopContext
from app.db import Conversation
from app.schemas import ChatHistoryOut, ChatMessage, ChatOut, PendingOrder


def thread_id(store_id: str, customer_id: str, chat_id: str) -> str:
    """Built by the server, so a customer can only ever reach their own chats in this store."""
    return f"{store_id}:{customer_id}:{chat_id}"


def _pending(interrupts) -> PendingOrder | None:
    for it in interrupts or []:
        value = getattr(it, "value", None)
        if isinstance(value, dict) and value.get("type") == "confirm_order":
            return PendingOrder(**{k: value[k] for k in PendingOrder.model_fields})
    return None


def _turn_messages(messages: list) -> list:
    """Messages after the customer's latest message."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1:]
    return messages


def _to_chat_out(out: dict) -> ChatOut:
    turn = _turn_messages(out["messages"])
    reply = next((m.text for m in reversed(turn) if isinstance(m, AIMessage) and m.text and not m.tool_calls), "")
    products, seen = [], set()
    for m in turn:
        if isinstance(m, ToolMessage) and m.name == "search_catalog" and isinstance(m.artifact, list):
            for p in m.artifact:
                if p["sku"] not in seen:
                    seen.add(p["sku"])
                    products.append(p)
    pending = _pending(out.get("__interrupt__"))
    if pending and not reply:
        reply = "Please confirm your order below."
    return ChatOut(reply=reply, route=out.get("route"), products=products, pending_order=pending)


def _log_conversation(tid: str, store_id: str, customer_id: str, text: str):
    with db.session() as s:
        conv = s.get(Conversation, tid) or Conversation(thread_id=tid, store_id=store_id, user_id=customer_id)
        conv.last_message, conv.turns = text[:200], (conv.turns or 0) + 1
        s.add(conv)
        s.commit()


def send(graph, store_id: str, customer_id: str, chat_id: str, message: str) -> ChatOut:
    tid = thread_id(store_id, customer_id, chat_id)
    cfg = {"configurable": {"thread_id": tid}}
    ctx = ShopContext(store_id=store_id, user_id=customer_id)
    if graph.get_state(cfg).interrupts:           # an order is waiting: a new message means "no"
        graph.invoke(Command(resume={"confirmed": False}), cfg, context=ctx)
    out = graph.invoke({"messages": [HumanMessage(message)]}, cfg, context=ctx)
    _log_conversation(tid, store_id, customer_id, message)
    return _to_chat_out(out)


def confirm(graph, store_id: str, customer_id: str, chat_id: str, confirmed: bool) -> ChatOut | None:
    """None if there is nothing to confirm."""
    tid = thread_id(store_id, customer_id, chat_id)
    cfg = {"configurable": {"thread_id": tid}}
    if not graph.get_state(cfg).interrupts:
        return None
    out = graph.invoke(Command(resume={"confirmed": confirmed}), cfg,
                       context=ShopContext(store_id=store_id, user_id=customer_id))
    return _to_chat_out(out)


def history(graph, tid: str) -> ChatHistoryOut:
    state = graph.get_state({"configurable": {"thread_id": tid}})
    messages = []
    for m in state.values.get("messages", []):
        if isinstance(m, HumanMessage):
            messages.append(ChatMessage(role="customer", text=m.text))
        elif isinstance(m, AIMessage) and m.text and not m.tool_calls:
            messages.append(ChatMessage(role="assistant", text=m.text))
    return ChatHistoryOut(messages=messages, pending_order=_pending(state.interrupts))
