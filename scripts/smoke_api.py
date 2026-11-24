"""End-to-end check over real HTTP against a running server (real Haiku, a few cents).
Start the server first:  uvicorn app.main:app --port 8765     Then: python -m scripts.smoke_api [base_url]"""
import secrets
import sys
import time

import httpx

sys.stdout.reconfigure(encoding="utf-8")
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
api = httpx.Client(base_url=BASE, timeout=120)

for _ in range(60):                                       # wait for the server to come up
    try:
        if api.get("/api/health").status_code == 200:
            break
    except httpx.TransportError:
        time.sleep(0.5)
else:
    sys.exit("Server did not start.")


def ok(r):
    if r.status_code >= 400:
        sys.exit(f"{r.request.method} {r.request.url} -> {r.status_code} {r.text}")
    return r.json()


# 1. a new seller signs up and uploads a catalog
email, password = f"seller-{secrets.token_hex(3)}@example.com", secrets.token_urlsafe(12)
signup = ok(api.post("/api/auth/signup", json={"email": email, "password": password, "store_name": "Hira Lawn Studio"}))
seller = {"Authorization": f"Bearer {signup['token']}"}
store_id = signup["store"]["id"]
print(f"seller signed up -> store '{store_id}'")
ok(api.put("/api/seller/settings", headers=seller,
           json={"name": "Hira Lawn Studio", "delivery_fees": {"islamabad": 150}, "default_delivery_fee": 300}))
csv = ("sku,name,category,color,price,stock,description,image_url\n"
       "HL-1,Mahnoor Embroidered Kurta,kurta,green,3500,S:2;M:3;L:1,Lawn kurta with neckline embroidery,\n"
       "HL-2,Organza Dupatta,dupatta,green,1500,6,Light organza dupatta,\n"
       "HL-3,Raw Silk Trouser,trouser,white,1800,S:2;M:2,,\n")
print("csv upload:", ok(api.post("/api/seller/products/upload", headers=seller, files={"file": ("c.csv", csv)})))

# 2. a customer chats with that store
cust = {"Authorization": f"Bearer {ok(api.post('/api/customer/session'))['customer_token']}"}


def say(text):
    r = ok(api.post(f"/api/stores/{store_id}/chat", headers=cust, json={"chat_id": "w1", "message": text}))
    print(f"\nCUSTOMER: {text}\n   route={r['route']} cards={[p['sku'] for p in r['products']]}\n   BOT: {r['reply']}")
    return r


say("Green kurta chahiye size M, budget 4000")
r = say("HL-1 size M ek piece, Islamabad. Naam Sana Ali, 03211234567, House 5, Street 9, F-10/3")
if not r["pending_order"]:
    r = say("Yes please place the order")
if r["pending_order"]:
    print(f"\n   [CONFIRM CARD] total Rs {r['pending_order']['total']} -> customer clicks YES")
    r = ok(api.post(f"/api/stores/{store_id}/chat/confirm", headers=cust, json={"chat_id": "w1", "confirmed": True}))
    print(f"   BOT: {r['reply']}")

# 3. the seller sees and approves it
orders = ok(api.get("/api/seller/orders", headers=seller, params={"status": "awaiting_approval"}))
print(f"\nseller sees {len(orders)} order(s) awaiting approval: {[(o['order_id'], o['total']) for o in orders]}")
if orders:
    print("approve:", ok(api.post(f"/api/seller/orders/{orders[0]['order_id']}/approve", headers=seller))["status"])
print("customer my-orders:", [(o["order_id"], o["status"]) for o in ok(api.get(f"/api/stores/{store_id}/my-orders", headers=cust))])
print("stats:", ok(api.get("/api/seller/stats", headers=seller)))
print("conversations:", ok(api.get("/api/seller/conversations", headers=seller)))
print("other store's chat as this seller ->",
      api.get("/api/seller/conversations/noor-threads:x:y", headers=seller).status_code)
