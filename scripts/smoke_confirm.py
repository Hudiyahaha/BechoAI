"""Real Haiku check that the Order agent goes straight to the confirm button (no 'reply yes' in chat).
Run: python -m scripts.smoke_confirm"""
import sys
import tempfile
from pathlib import Path

from langchain.messages import HumanMessage

from app import db
from app.agent import build_graph, sqlite_memory
from app.chat import _to_chat_out
from app.context import ShopContext
from app.seed import seed

sys.stdout.reconfigure(encoding="utf-8")
tmp = Path(tempfile.mkdtemp())
db.configure(f"sqlite:///{tmp / 'shop.db'}")
seed()
saver, store = sqlite_memory(str(tmp / "memory.db"))
graph = build_graph(checkpointer=saver, store=store)
ctx, cfg = ShopContext("noor-threads", "c-1"), {"configurable": {"thread_id": "t"}}

for text in ["NT-201 size M ek piece chahiye, Rawalpindi",
             "Naam Sana Ali, 03211234567, House 5, Street 9, Satellite Town"]:
    r = _to_chat_out(graph.invoke({"messages": [HumanMessage(text)]}, cfg, context=ctx))
    shown = f"yes (total Rs {r.pending_order.total})" if r.pending_order else "no"
    print(f"\nCUSTOMER: {text}\n   BOT: {r.reply}\n   confirm button shown: {shown}")
