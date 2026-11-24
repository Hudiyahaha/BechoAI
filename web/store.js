// Customer store page: catalog + chat with the BechoAI sales agent.
const storeId = decodeURIComponent(location.pathname.split("/")[2] || "");
const $ = (id) => document.getElementById(id);
const state = { token: null, chatId: null, products: [], busy: false };

// ── identity: anonymous, server-signed customer token kept on this device ──
async function ensureToken(force = false) {
  state.token = force ? null : storage.get("bechoai_customer");
  if (!state.token) {
    state.token = (await api("/api/customer/session", { method: "POST" })).customer_token;
    storage.set("bechoai_customer", state.token);
  }
  return state.token;
}

async function customerApi(path, options = {}) {
  await ensureToken();
  try {
    return await api(path, { ...options, token: state.token });
  } catch (err) {
    if (err.status !== 401) throw err;
    await ensureToken(true);                       // token expired or server secret changed
    return api(path, { ...options, token: state.token });
  }
}

function newChatId() {
  const id = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2)).replace(/[^A-Za-z0-9-]/g, "");
  storage.set(`bechoai_chat_${storeId}`, id);
  return id;
}

// ── catalog ──
function productCard(p) {
  const sizes = sizesText(p.sizes_in_stock).map((s) => `<span>${escapeHtml(s)}</span>`).join("");
  return `<button class="product card" data-sku="${escapeHtml(p.sku)}">
    ${swatchHtml(p)}
    <div class="info"><h3>${escapeHtml(p.name)}</h3>
      <div class="muted" style="font-size:13px">${escapeHtml(p.color)} · ${escapeHtml(p.category)}</div>
      <div class="price">${rs(p.price)}</div><div class="sizes">${sizes}</div></div></button>`;
}

async function loadProducts() {
  const params = new URLSearchParams();
  for (const [key, id] of [["query", "f-query"], ["category", "f-category"], ["color", "f-color"], ["size", "f-size"], ["max_price", "f-max"]]) {
    const value = $(id).value.trim();
    if (value) params.set(key, value);
  }
  const products = await api(`/api/stores/${encodeURIComponent(storeId)}/products?${params}`);
  products.forEach((p) => { state.products[p.sku] = p; });
  $("grid").innerHTML = products.length ? products.map(productCard).join("")
    : `<div class="empty card" style="grid-column:1/-1">Nothing matches. Try fewer filters, or ask the assistant.</div>`;
  return products;
}

function fillFilters(products) {
  const add = (id, values) => [...new Set(values)].sort().forEach((v) => $(id).insertAdjacentHTML("beforeend", `<option>${escapeHtml(v)}</option>`));
  add("f-category", products.map((p) => p.category));
  add("f-color", products.map((p) => p.color));
}

function openProduct(sku) {
  const p = state.products[sku];
  if (!p) return;
  const sizes = Object.entries(p.sizes_in_stock).map(([s, n]) => `<span>${s === "ONE SIZE" ? "One size" : escapeHtml(s)}: ${n} left</span>`).join("");
  $("product-body").innerHTML = `<button class="btn small close" data-close>✕</button>
    ${swatchHtml(p, true)}
    <h2 style="margin-top:14px">${escapeHtml(p.name)}</h2>
    <div class="muted">${escapeHtml(p.sku)} · ${escapeHtml(p.color)} · ${escapeHtml(p.category)}</div>
    <p class="price" style="font-size:20px;margin:8px 0">${rs(p.price)}</p>
    <p>${escapeHtml(p.description || "")}</p>
    <div class="sizes">${sizes}</div>
    <div style="margin-top:16px;display:flex;gap:8px">
      <button class="btn primary" data-ask="${escapeHtml(p.sku)}">Ask the assistant / order</button></div>`;
  $("product-dialog").showModal();
}

// ── chat ──
function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (role === "assistant") div.innerHTML = renderMarkdown(text); else div.textContent = text;
  $("log").appendChild(div);
  $("log").scrollTop = $("log").scrollHeight;
  return div;
}

function addProductChips(products) {
  if (!products?.length) return;
  products.forEach((p) => { state.products[p.sku] = p; });
  const row = document.createElement("div");
  row.className = "chips";
  row.innerHTML = products.map((p) => `<button class="chip" data-sku="${escapeHtml(p.sku)}">
      <span class="dot" style="background:${colorFor(p.color)}"></span>
      <span><strong>${escapeHtml(p.sku)}</strong> ${escapeHtml(p.name)}<br><span class="muted">${rs(p.price)}</span></span></button>`).join("");
  $("log").appendChild(row);
}

function addConfirmButtons() {
  const row = document.createElement("div");
  row.className = "confirm";
  row.innerHTML = `<div class="row" style="margin:0"><button class="btn primary" data-confirm="yes">Yes, confirm order</button>
    <button class="btn" data-confirm="no">No, cancel</button></div>`;
  $("log").appendChild(row);
  $("log").scrollTop = $("log").scrollHeight;
}

function showReply(out) {
  addMessage("assistant", out.reply || "…");
  addProductChips(out.products);
  if (out.pending_order) addConfirmButtons();
}

async function withTyping(promise) {
  state.busy = true;
  $("send").disabled = true;
  const typing = addMessage("assistant", "Typing…");
  typing.classList.add("typing");
  try { return await promise; }
  catch (err) { return { reply: `Sorry, something went wrong: ${err.message}` }; }
  finally { typing.remove(); state.busy = false; $("send").disabled = false; }
}

async function send(text) {
  text = text.trim();
  if (!text || state.busy) return;
  $("suggest").classList.add("hidden");
  document.querySelectorAll(".confirm button").forEach((b) => { b.disabled = true; });
  addMessage("customer", text);
  showReply(await withTyping(customerApi(`/api/stores/${encodeURIComponent(storeId)}/chat`,
    { method: "POST", body: { chat_id: state.chatId, message: text } })));
}

function confirmOrder(yes, card) {
  card.querySelectorAll("button").forEach((b) => { b.disabled = true; });
  send(yes ? "yes" : "no");                         // same as typing it: the agent asks for YES / NO
}

async function loadHistory() {
  const hist = await customerApi(`/api/stores/${encodeURIComponent(storeId)}/chat/${state.chatId}`);
  $("log").innerHTML = "";
  if (!hist.messages.length) {
    addMessage("assistant", "Assalam o Alaikum! I can help you find something, check sizes and place a cash-on-delivery order. What are you looking for?");
    return;
  }
  $("suggest").classList.add("hidden");
  hist.messages.forEach((m) => addMessage(m.role, m.text));
  if (hist.pending_order) addConfirmButtons();
}

// ── orders ──
async function showOrders() {
  $("orders-list").innerHTML = `<p class="muted">Loading…</p>`;
  $("orders-dialog").showModal();
  const orders = await customerApi(`/api/stores/${encodeURIComponent(storeId)}/my-orders`);
  const label = { awaiting_approval: "Waiting for seller", approved: "Approved", rejected: "Rejected" };
  $("orders-list").innerHTML = orders.length ? orders.map((o) => `<div class="order">
      <div style="display:flex;justify-content:space-between;gap:8px"><strong>Order #${o.order_id}</strong>
        <span class="badge ${o.status}">${label[o.status] || escapeHtml(o.status)}</span></div>
      <div class="muted" style="font-size:13px">${escapeHtml(o.created_at)} · ${rs(o.total)}</div>
      <div style="font-size:14px">${o.items.map((i) => `${i.qty} × ${escapeHtml(i.name)} (${escapeHtml(i.size)})`).join(", ")}</div></div>`).join("")
    : `<p class="muted">No orders yet. Ask the assistant to place one.</p>`;
}

// ── wiring ──
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-sku],[data-close],[data-ask],[data-confirm]");
  if (!t) return;
  if (t.dataset.close !== undefined) t.closest("dialog").close();
  else if (t.dataset.ask) {
    $("product-dialog").close();
    $("chat").classList.add("open");
    $("input").value = `I'm interested in ${t.dataset.ask}. `;
    $("input").focus();
  } else if (t.dataset.confirm) confirmOrder(t.dataset.confirm === "yes", t.closest(".confirm"));
  else if (t.dataset.sku) openProduct(t.dataset.sku);
});
document.querySelectorAll("dialog").forEach((d) => d.addEventListener("click", (e) => { if (e.target === d) d.close(); }));
$("composer").addEventListener("submit", (e) => { e.preventDefault(); const text = $("input").value; $("input").value = ""; send(text); });
$("suggest").addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") send(e.target.textContent); });
$("orders-btn").addEventListener("click", () => showOrders().catch((err) => { $("orders-list").textContent = err.message; }));
$("chat-fab").addEventListener("click", () => { $("chat").classList.add("open"); $("input").focus(); });
$("chat-close").addEventListener("click", () => $("chat").classList.remove("open"));
["f-category", "f-color", "f-size"].forEach((id) => $(id).addEventListener("change", loadProducts));
["f-query", "f-max"].forEach((id) => $(id).addEventListener("input", debounce(loadProducts, 300)));

const newChat = document.createElement("button");
newChat.className = "btn small";
newChat.textContent = "New chat";
newChat.addEventListener("click", () => { state.chatId = newChatId(); $("suggest").classList.remove("hidden"); loadHistory(); });
$("chat-close").before(newChat);

(async function init() {
  try {
    const store = await api(`/api/stores/${encodeURIComponent(storeId)}`);
    document.title = `${store.name} · BechoAI`;
    $("store-name").textContent = store.name;
    $("hero-title").textContent = `Welcome to ${store.name}`;
    fillFilters(await loadProducts());
    state.chatId = storage.get(`bechoai_chat_${storeId}`) || newChatId();
    await loadHistory();
  } catch (err) {
    $("hero-title").textContent = err.status === 404 ? "Store not found" : "Something went wrong";
    $("grid").innerHTML = `<div class="empty card" style="grid-column:1/-1">${escapeHtml(err.message)}</div>`;
  }
})();
