"""The graded agent API: POST /chat and GET /health, in the exact hackathon format."""
import pytest
from fastapi.testclient import TestClient
from langchain.messages import AIMessage

from app import db, services
from app.chat import answer_kind, thread_id
from app.main import app
from app.memory import namespace
from app.context import ShopContext
from tests.conftest import STORE
from tests.test_api import set_graph
from tests.test_middleware import ORDER, tool_call


@pytest.fixture
def client():
    set_graph([])
    with TestClient(app) as c:
        yield c
    app.state.graph = None


def chat(client, message, user="u1", thread="t1"):
    r = client.post("/chat", json={"user_id": user, "thread_id": thread, "message": message})
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.text == "ok"


def test_chat_format_exactly(client):
    set_graph([tool_call("search_catalog", {"color": "red"}, 1), AIMessage("Red items: NT-201.")])
    out = chat(client, "red kurta?")
    assert set(out) == {"reply", "tool_calls", "tokens"} and set(out["tokens"]) == {"in", "out"}
    assert out["reply"] == "Red items: NT-201."
    assert out["tool_calls"] == [{"name": "search_catalog", "args": {"color": "red"}}]


def test_chat_needs_no_login_but_validates_body(client):
    assert client.post("/chat", json={"user_id": "u", "thread_id": "t"}).status_code == 422
    assert client.post("/chat", json={"user_id": "", "thread_id": "t", "message": "hi"}).status_code == 422


def test_order_confirmed_by_typing_yes(client):
    set_graph([tool_call("place_order", ORDER, 1), AIMessage("Order #1 is waiting for the seller.")])
    first = chat(client, "buy NT-202 size M, Ayesha Khan 03001234567 House 12 Street 4 G-11/2 Islamabad")
    assert "Reply **YES** to confirm" in first["reply"] and first["tool_calls"][0]["name"] == "place_order"
    with db.session() as s:
        assert services.get_orders(s, STORE, "u1") == []                # nothing saved before YES
    assert chat(client, "yes")["reply"] == "Order #1 is waiting for the seller."
    with db.session() as s:
        assert services.get_orders(s, STORE, "u1")[0]["status"] == "awaiting_approval"


def test_order_cancelled_by_no(client):
    set_graph([tool_call("place_order", ORDER, 1), AIMessage("Okay, I didn't place it.")])
    chat(client, "buy NT-202")
    assert chat(client, "no thanks")["reply"] == "Okay, I didn't place it."
    with db.session() as s:
        assert services.get_orders(s, STORE, "u1") == []


def test_other_message_cancels_and_is_answered(client):
    set_graph([tool_call("place_order", ORDER, 1), AIMessage("Not placed."), AIMessage("Here are blue items.")])
    chat(client, "buy NT-202")
    assert chat(client, "actually show me blue ones")["reply"] == "Here are blue items."
    with db.session() as s:
        assert services.get_orders(s, STORE, "u1") == []


@pytest.mark.parametrize("text,kind", [
    ("yes", "yes"), ("Yes!", "yes"), ("haan ji", "yes"), ("ok place it", "yes"), ("confirm", "yes"),
    ("no", "no"), ("nahi", "no"), ("cancel it", "no"),
    ("yes but change the size to L", "other"), ("what is the delivery time?", "other"), ("", "other"),
])
def test_answer_kind(text, kind):
    assert answer_kind(text) == kind


def test_users_and_threads_are_isolated(client):
    set_graph([AIMessage("A's reply"), AIMessage("B's reply")])
    chat(client, "my secret is pastel", user="alice", thread="t1")
    chat(client, "hello", user="bob", thread="t1")                     # same thread_id, other user
    graph = app.state.graph
    a = graph.get_state({"configurable": {"thread_id": thread_id(STORE, "alice", "t1")}}).values["messages"]
    b = graph.get_state({"configurable": {"thread_id": thread_id(STORE, "bob", "t1")}}).values["messages"]
    assert [m.text for m in b] == ["hello", "B's reply"] and "pastel" not in str([m.text for m in b])
    assert len(a) == 2


def test_ids_with_separators_cannot_collide():
    assert thread_id(STORE, "a:b", "c") != thread_id(STORE, "a", "b:c")
    assert namespace(ShopContext(STORE, "a.b")) != namespace(ShopContext(STORE, "a"))
    assert all("." not in label for label in namespace(ShopContext(STORE, "user.name@x.com")))


def test_website_is_served(client):
    assert client.get("/s/noor-threads").status_code == 200
    assert "BECHOAI_API" in client.get("/config.js").text


def test_tool_calls_reported_while_waiting_and_after_yes(client):
    set_graph([tool_call("calculate_total", {"items": ORDER["items"], "city": "Islamabad"}, 1),
               tool_call("place_order", ORDER, 2), AIMessage("Placed, waiting for the seller.")])
    first = chat(client, "buy NT-202 M")
    assert [c["name"] for c in first["tool_calls"]] == ["calculate_total", "place_order"]
    second = chat(client, "yes")
    assert [c["name"] for c in second["tool_calls"]] == ["place_order"]
    assert second["tool_calls"][0]["args"]["city"] == "Islamabad"
