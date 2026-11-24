"""Runs the real tools inside a tiny LangGraph graph (no LLM, no API key): a fake 'AI' node issues one tool call."""
from langchain.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from app import db, services
from app.context import ShopContext
from app.tools import TOOLS
from tests.conftest import STORE

ORDER_ARGS = {"items": [{"sku": "NT-201", "size": "M", "qty": 1}], "customer_name": "Ayesha Khan",
              "phone": "03001234567", "address": "House 12, Street 4, G-11/2", "city": "Islamabad"}


def run_tool(name, args, user="user-a", store=STORE, thread="t1", graph=None):
    if graph is None:
        graph = build_graph(name, args)
    cfg = {"configurable": {"thread_id": thread}}
    out = graph.invoke({"messages": []}, cfg, context=ShopContext(store_id=store, user_id=user))
    return graph, cfg, out


def build_graph(name, args):
    def fake_ai(state):
        return {"messages": [AIMessage("", tool_calls=[{"name": name, "args": args, "id": "call-1"}])]}
    g = StateGraph(MessagesState, context_schema=ShopContext)
    g.add_node("ai", fake_ai)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "ai")
    g.add_edge("ai", "tools")
    g.add_edge("tools", END)
    return g.compile(checkpointer=InMemorySaver())


def last(out):
    return out["messages"][-1]


def test_search_tool_uses_context_store():
    _, _, out = run_tool("search_catalog", {"color": "red", "max_price": 5000})
    msg = last(out)
    assert "NT-201" in msg.content and "OS-1" not in msg.content
    assert [p["sku"] for p in msg.artifact] == ["NT-401", "NT-201", "NT-101"]   # structured data for the UI
    _, _, out = run_tool("search_catalog", {"color": "red"}, store="other-store")
    assert "OS-1" in last(out).content and "NT-" not in last(out).content


def test_calculate_total_tool():
    _, _, out = run_tool("calculate_total", {"items": ORDER_ARGS["items"], "city": "Islamabad"})
    assert "Total (cash on delivery): Rs 3,400" in last(out).content


def test_place_order_waits_for_customer_then_seller():
    graph, cfg, out = run_tool("place_order", ORDER_ARGS)
    assert out["__interrupt__"][0].value["total"] == 3400                       # paused, nothing saved yet
    with db.session() as s:
        assert services.get_orders(s, STORE, "user-a") == []
    out = graph.invoke(Command(resume={"confirmed": True}), cfg, context=ShopContext(STORE, "user-a"))
    assert "waiting for the seller's approval" in last(out).content
    with db.session() as s:
        [order] = services.get_orders(s, STORE, "user-a")
    assert order["status"] == "awaiting_approval"


def test_place_order_declined_saves_nothing():
    graph, cfg, _ = run_tool("place_order", ORDER_ARGS)
    out = graph.invoke(Command(resume={"confirmed": False}), cfg, context=ShopContext(STORE, "user-a"))
    assert "No order was placed" in last(out).content
    with db.session() as s:
        assert services.get_orders(s, STORE, "user-a") == []


def test_place_order_bad_phone_never_asks_to_confirm():
    _, _, out = run_tool("place_order", {**ORDER_ARGS, "phone": "123"})
    assert "__interrupt__" not in out and "valid Pakistani mobile" in last(out).content


def test_order_status_is_per_user():
    graph, cfg, _ = run_tool("place_order", ORDER_ARGS)
    graph.invoke(Command(resume={"confirmed": True}), cfg, context=ShopContext(STORE, "user-a"))
    _, _, out = run_tool("get_order_status", {"order_id": 1}, user="user-a", thread="t2")
    assert "Order #1 | awaiting_approval" in last(out).content
    _, _, out = run_tool("get_order_status", {"order_id": 1}, user="user-b", thread="t3")
    assert "couldn't find order #1" in last(out).content


def test_search_falls_back_when_words_match_nothing():
    _, _, out = run_tool("search_catalog", {"query": "new arrival", "color": "red", "size": "M"})
    msg = last(out)
    assert msg.content.startswith("No product mentions 'new arrival'") and "NT-201" in msg.content
