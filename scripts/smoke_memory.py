"""Real Haiku check of long-term memory across two chats, on throwaway databases. Costs about a cent.
Run: python -m scripts.smoke_memory"""
import sys
import tempfile
from pathlib import Path

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from app import db
from app.agent import build_graph, sqlite_memory
from app.context import ShopContext
from app.memory import namespace
from app.seed import seed

sys.stdout.reconfigure(encoding="utf-8")
tmp = Path(tempfile.mkdtemp())
db.configure(f"sqlite:///{tmp / 'shop.db'}")
seed()
saver, store = sqlite_memory(str(tmp / "memory.db"))
graph = build_graph(checkpointer=saver, store=store)
ctx = ShopContext(store_id="noor-threads", user_id="customer-1")


def show(out, seen):
    """Print messages not printed before (tracked by id: summarising removes old messages)."""
    for m in [m for m in out["messages"] if m.id not in seen and not isinstance(m, HumanMessage)]:
        seen.add(m.id)
        if isinstance(m, ToolMessage):
            print(f"   [tool {m.name}] {m.content[:160]!r}")
        elif isinstance(m, AIMessage) and m.tool_calls:
            print(f"   [calls] {[(c['name'], c['args']) for c in m.tool_calls]}")
        elif isinstance(m, AIMessage):
            print(f"   BOT: {m.text}")
    return seen


for thread, turns in [("chat-1", ["Main hamesha size M pehenti hoon, aur mujhe red colour pasand hai. Kurta dikhao"]),
                      ("chat-2 (next day)", ["Kuch naya dikhao", "what do you remember about me?", "please forget me"])]:
    print(f"\n===== {thread} =====")
    seen = set()
    for text in turns:
        print(f"\nCUSTOMER: {text}")
        out = graph.invoke({"messages": [HumanMessage(text)]}, {"configurable": {"thread_id": thread}}, context=ctx)
        print(f"   route={out.get('route')}")
        seen = show(out, seen)
    print(f"\n   saved facts now: {[(i.value['kind'], i.value['value']) for i in store.search(namespace(ctx))]}")
