"""Runs the LangGraph agent for one message and turns its state into API responses.

Order confirmation happens in the conversation itself: when the agent prepares an order, the reply shows the
summary and asks "Reply YES to confirm or NO to cancel". The next message resumes the paused graph with that
answer. (The graded POST /chat API has no buttons; the website's buttons simply send "yes" / "no".)
"""
import re
from dataclasses import dataclass, field
from urllib.parse import quote

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.types import Command

from app import db
from app.context import ShopContext
from app.db import Conversation
from app.schemas import ChatHistoryOut, ChatMessage, PendingOrder

YES_WORDS = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "confirm", "confirmed", "sure", "haan", "han", "ha",
             "ji", "jee", "g", "theek", "thik", "done", "proceed", "place"}
NO_WORDS = {"no", "n", "nope", "nah", "cancel", "nahi", "nahin", "nai", "mat", "stop", "dont", "don't"}
HEDGES = re.compile(r"\b(but|change|instead|wait|ruko|lekin|magar|not|nahi|no|cancel)\b")


def thread_id(store_id: str, customer_id: str, chat_id: str) -> str:
    """Built by the server and escaped, so no user_id/thread_id combination can reach another user's chat."""
    return ":".join(quote(part, safe="") for part in (store_id, customer_id, chat_id))


def answer_kind(message: str) -> str:
    """'yes', 'no' or 'other' for a reply to the confirmation question."""
    words = re.findall(r"[a-z']+", message.lower())
    if not words or len(words) > 6:
        return "other"
    if words[0] in NO_WORDS:
        return "no"
    if words[0] in YES_WORDS and not HEDGES.search(" ".join(words[1:])):
        return "yes"
    return "other"


@dataclass
class Turn:
    reply: str
    route: str | None = None
    tool_calls: list = field(default_factory=list)
    products: list = field(default_factory=list)
    pending_order: PendingOrder | None = None
    tokens_in: int = 0
    tokens_out: int = 0


def _pending(interrupts) -> PendingOrder | None:
    for it in interrupts or []:
        value = getattr(it, "value", None)
        if isinstance(value, dict) and value.get("type") == "confirm_order":
            return PendingOrder(**{k: value[k] for k in PendingOrder.model_fields})
    return None


def confirm_text(p: PendingOrder) -> str:
    return (f"Please check your order:\n{p.summary}\nDeliver to: {p.customer_name}, {p.phone}, {p.address}, {p.city}\n\n"
            "Reply **YES** to confirm or **NO** to cancel.")


def _turn_messages(messages: list) -> list:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1:]
    return messages


def _paused_messages(graph, cfg) -> list:
    """Messages inside an agent subgraph that is paused (waiting for YES / NO); not in the main state yet."""
    msgs = []
    for task in graph.get_state(cfg, subgraphs=True).tasks:
        sub = getattr(task, "state", None)
        if sub is not None and hasattr(sub, "values"):
            msgs += sub.values.get("messages", [])
    return msgs


def _all_ids(graph, cfg) -> set:
    return {m.id for m in graph.get_state(cfg).values.get("messages", []) + _paused_messages(graph, cfg)}


def _collect(out: dict, turn: Turn, seen_ids: set, graph=None, cfg=None):
    """Add this invocation's tool calls / products / reply to the turn (skipping messages seen before)."""
    all_msgs = out["messages"] + (_paused_messages(graph, cfg) if out.get("__interrupt__") and graph else [])
    msgs = [m for m in _turn_messages(all_msgs) if m.id not in seen_ids]
    for m in msgs:
        if isinstance(m, AIMessage):
            turn.tool_calls += [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
        if isinstance(m, ToolMessage) and m.name == "search_catalog" and isinstance(m.artifact, list):
            known = {p["sku"] for p in turn.products}
            turn.products += [p for p in m.artifact if p["sku"] not in known]
    reply = next((m.text for m in reversed(msgs) if isinstance(m, AIMessage) and m.text and not m.tool_calls), "")
    turn.pending_order = _pending(out.get("__interrupt__"))
    if turn.pending_order:
        turn.reply = confirm_text(turn.pending_order)
    elif reply:
        turn.reply = reply
    turn.route = out.get("route", turn.route)


def _log_conversation(tid: str, store_id: str, customer_id: str, text: str):
    with db.session() as s:
        conv = s.get(Conversation, tid) or Conversation(thread_id=tid, store_id=store_id, user_id=customer_id)
        conv.last_message, conv.turns = text[:200], (conv.turns or 0) + 1
        s.add(conv)
        s.commit()


def run_turn(graph, store_id: str, customer_id: str, chat_id: str, message: str) -> Turn:
    tid = thread_id(store_id, customer_id, chat_id)
    cfg = {"configurable": {"thread_id": tid}}
    ctx = ShopContext(store_id=store_id, user_id=customer_id)
    turn = Turn(reply="")
    seen = _all_ids(graph, cfg)
    with get_usage_metadata_callback() as usage:
        snapshot = graph.get_state(cfg)
        if snapshot.interrupts:                                # an order is waiting for YES / NO
            kind = answer_kind(message)
            pending = next((i.value for i in snapshot.interrupts if isinstance(i.value, dict)), {})
            if kind == "yes":                                  # place_order really runs now
                turn.tool_calls.append({"name": "place_order", "args": pending.get("args", {})})
            out = graph.invoke(Command(resume={"confirmed": kind == "yes"}), cfg, context=ctx)
            _collect(out, turn, seen, graph, cfg)
            seen |= _all_ids(graph, cfg)
            if kind == "other":                                # cancelled; now handle the new message
                out = graph.invoke({"messages": [HumanMessage(message)]}, cfg, context=ctx)
                _collect(out, turn, seen, graph, cfg)
        else:
            out = graph.invoke({"messages": [HumanMessage(message)]}, cfg, context=ctx)
            _collect(out, turn, seen, graph, cfg)
    for u in usage.usage_metadata.values():
        turn.tokens_in += u.get("input_tokens", 0)
        turn.tokens_out += u.get("output_tokens", 0)
    turn.reply = turn.reply or "Sorry, I couldn't answer that. Could you rephrase?"
    _log_conversation(tid, store_id, customer_id, message)
    return turn


def history(graph, tid: str) -> ChatHistoryOut:
    state = graph.get_state({"configurable": {"thread_id": tid}})
    messages = []
    for m in state.values.get("messages", []):
        if isinstance(m, HumanMessage):
            messages.append(ChatMessage(role="customer", text=m.text))
        elif isinstance(m, AIMessage) and m.text and not m.tool_calls:
            messages.append(ChatMessage(role="assistant", text=m.text))
    pending = _pending(state.interrupts)
    if pending:
        messages.append(ChatMessage(role="assistant", text=confirm_text(pending)))
    return ChatHistoryOut(messages=messages, pending_order=pending)
