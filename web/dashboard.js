// Seller dashboard: login/signup, overview, orders (approve/reject), products (CRUD + CSV), chats, settings.
const TOKEN_KEY = "bechoai_seller";
const $ = (id) => document.getElementById(id);
let token = storage.get(TOKEN_KEY);
let store = null;
let signupMode = false;
let products = [];

const call = (path, opts = {}) => api(path, { ...opts, token }).catch((err) => {
  if (err.status === 401) { logout(); }
  throw err;
});

// ── auth ──
function showAuth() {
  $("auth").classList.remove("hidden");
  $("dash").classList.add("hidden");
  $("top-actions").classList.add("hidden");
}

function setMode(signup) {
  signupMode = signup;
  $("auth-title").textContent = signup ? "Create your store" : "Log in";
  $("auth-submit").textContent = signup ? "Create store" : "Log in";
  $("store-name-field").classList.toggle("hidden", !signup);
  $("a-store").required = signup;
  $("a-pass").autocomplete = signup ? "new-password" : "current-password";
  $("auth-switch-text").textContent = signup ? "Already selling with us?" : "New seller?";
  $("auth-switch").textContent = signup ? "Log in" : "Create a store (14-day free trial)";
  $("auth-error").textContent = "";
}

$("auth-switch").addEventListener("click", (e) => { e.preventDefault(); setMode(!signupMode); });

$("auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("auth-error").textContent = "";
  $("auth-submit").disabled = true;
  try {
    const body = { email: $("a-email").value.trim(), password: $("a-pass").value };
    if (signupMode) body.store_name = $("a-store").value.trim();
    const out = await api(signupMode ? "/api/auth/signup" : "/api/auth/login", { method: "POST", body });
    token = out.token;
    storage.set(TOKEN_KEY, token);
    $("a-pass").value = "";
    await start();
  } catch (err) {
    $("auth-error").textContent = err.message;
  } finally {
    $("auth-submit").disabled = false;
  }
});

function logout() {
  token = null;
  store = null;
  storage.remove(TOKEN_KEY);
  showAuth();
}
$("logout").addEventListener("click", logout);

// ── tabs ──
const loaders = { overview: loadOverview, orders: loadOrders, products: loadProducts, chats: loadChats, settings: loadSettings };
function showTab(name) {
  document.querySelectorAll(".side button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll("[data-panel]").forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== name));
  storage.set("bechoai_tab", name);
  loaders[name]().catch((err) => alert(err.message));
}
document.querySelectorAll(".side button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

async function start() {
  if (!token) return showAuth();
  try {
    store = await call("/api/seller/me");
  } catch {
    return showAuth();
  }
  $("auth").classList.add("hidden");
  $("dash").classList.remove("hidden");
  $("top-actions").classList.remove("hidden");
  $("store-title").textContent = store.name;
  $("store-link").href = `/s/${encodeURIComponent(store.id)}`;
  const tab = storage.get("bechoai_tab");
  showTab(loaders[tab] ? tab : "overview");
}

// ── overview ──
async function loadOverview() {
  const s = await call("/api/seller/stats");
  const waiting = s.orders_by_status.awaiting_approval || 0;
  updatePendingBadge(waiting);
  const kpis = [
    [rs(s.revenue), "Revenue (approved orders)"],
    [waiting, `Waiting for approval · ${rs(s.pending_value)}`],
    [s.conversations, "Customer conversations"],
    [`${Math.round(s.chat_to_order_rate * 100)}%`, "Chats that became orders"],
  ];
  $("kpis").innerHTML = kpis.map(([v, l]) => `<div class="card kpi"><div class="v">${escapeHtml(v)}</div><div class="l">${escapeHtml(l)}</div></div>`).join("");
  $("top-products").innerHTML = s.top_products.length
    ? `<table><tbody>${s.top_products.map((p) => `<tr><td>${escapeHtml(p.sku)}</td><td>${escapeHtml(p.name)}</td><td>${p.qty} sold</td><td>${rs(p.revenue)}</td></tr>`).join("")}</tbody></table>`
    : `<p class="muted">No approved orders yet.</p>`;
  $("low-stock").innerHTML = s.low_stock.length
    ? `<table><tbody>${s.low_stock.map((p) => `<tr><td>${escapeHtml(p.sku)}</td><td>${escapeHtml(p.name)}</td><td>${p.left} left</td></tr>`).join("")}</tbody></table>`
    : `<p class="muted">Everything is well stocked.</p>`;
}

function updatePendingBadge(n) {
  $("pending-count").textContent = n;
  $("pending-count").classList.toggle("hidden", !n);
}

// ── orders ──
const STATUS = { awaiting_approval: "Waiting", approved: "Approved", rejected: "Rejected" };

async function loadOrders() {
  const filter = $("order-filter").value;
  const orders = await call(`/api/seller/orders${filter ? `?status=${encodeURIComponent(filter)}` : ""}`);
  $("orders-body").innerHTML = orders.length ? orders.map((o) => `
    <tr>
      <td>${o.order_id}</td>
      <td>${escapeHtml(o.created_at)}</td>
      <td><strong>${escapeHtml(o.customer_name)}</strong><br><span class="muted">${escapeHtml(o.phone)}<br>${escapeHtml(o.address)}, ${escapeHtml(o.city)}</span></td>
      <td>${o.items.map((i) => `${i.qty} × ${escapeHtml(i.name)} (${escapeHtml(i.sku)}, ${escapeHtml(i.size)})`).join("<br>")}</td>
      <td>${rs(o.total)}<br><span class="muted" style="font-size:12px">incl. ${rs(o.delivery_fee)} delivery</span></td>
      <td><span class="badge ${escapeHtml(o.status)}">${STATUS[o.status] || escapeHtml(o.status)}</span></td>
      <td style="white-space:nowrap">${o.status === "awaiting_approval"
        ? `<button class="btn small primary" data-approve="${o.order_id}">Approve</button> <button class="btn small danger" data-reject="${o.order_id}">Reject</button>` : ""}</td>
    </tr>`).join("") : `<tr><td colspan="7" class="empty">No orders here.</td></tr>`;
  if (filter === "awaiting_approval") updatePendingBadge(orders.length);
}

$("order-filter").addEventListener("change", () => loadOrders().catch((err) => alert(err.message)));
$("orders-refresh").addEventListener("click", () => loadOrders().catch((err) => alert(err.message)));
$("orders-body").addEventListener("click", async (e) => {
  const id = e.target.dataset.approve || e.target.dataset.reject;
  if (!id) return;
  const approve = Boolean(e.target.dataset.approve);
  if (!approve && !confirm(`Reject order #${id}? The stock goes back to your catalog.`)) return;
  e.target.disabled = true;
  try {
    await call(`/api/seller/orders/${id}/${approve ? "approve" : "reject"}`, { method: "POST" });
    await loadOrders();
  } catch (err) {
    alert(err.message);
    e.target.disabled = false;
  }
});

// ── products ──
const stockText = (stock) => Object.entries(stock || {}).map(([k, v]) => `${k}:${v}`).join(", ");

function parseStock(text) {
  const stock = {};
  for (const part of text.split(/[,;]/)) {
    if (!part.trim()) continue;
    const m = part.match(/^\s*([^:]+?)\s*:\s*(\d+)\s*$/);
    if (!m) throw new Error(`Stock "${part.trim()}" should look like M:5`);
    stock[m[1].toUpperCase()] = Number(m[2]);
  }
  if (!Object.keys(stock).length) throw new Error("Add stock for at least one size.");
  return stock;
}

async function loadProducts() {
  products = await call("/api/seller/products");
  $("products-body").innerHTML = products.length ? products.map((p) => {
    const total = Object.values(p.stock).reduce((a, b) => a + b, 0);
    return `<tr>
      <td><strong>${escapeHtml(p.sku)}</strong></td>
      <td>${escapeHtml(p.name)}</td>
      <td>${escapeHtml(p.category)}</td>
      <td>${escapeHtml(p.color)}</td>
      <td>${rs(p.price)}</td>
      <td>${total ? escapeHtml(stockText(p.stock)) : '<span class="badge rejected">Out of stock</span>'}</td>
      <td style="white-space:nowrap"><button class="btn small" data-edit="${escapeHtml(p.sku)}">Edit</button> <button class="btn small danger" data-delete="${escapeHtml(p.sku)}">Delete</button></td>
    </tr>`;
  }).join("") : `<tr><td colspan="7" class="empty">No products yet. Add one or upload a CSV.</td></tr>`;
}

let editingSku = null;
function openProduct(p) {
  editingSku = p ? p.sku : null;
  $("pd-title").textContent = p ? `Edit ${p.sku}` : "Add product";
  $("p-sku").value = p?.sku || "";
  $("p-sku").readOnly = Boolean(p);
  $("p-name").value = p?.name || "";
  $("p-category").value = p?.category || "";
  $("p-color").value = p?.color || "";
  $("p-price").value = p?.price || "";
  $("p-stock").value = p ? stockText(p.stock) : "";
  $("p-desc").value = p?.description || "";
  $("p-image").value = p?.image_url || "";
  $("product-error").textContent = "";
  $("product-dialog").showModal();
}

$("add-product").addEventListener("click", () => openProduct(null));
$("products-body").addEventListener("click", async (e) => {
  const { edit, delete: del } = e.target.dataset;
  if (edit) openProduct(products.find((p) => p.sku === edit));
  if (del && confirm(`Delete ${del}? If it was ever ordered, it is kept but set to out of stock.`)) {
    try {
      const out = await call(`/api/seller/products/${encodeURIComponent(del)}`, { method: "DELETE" });
      if (out.result === "archived") alert(`${del} has orders, so it was set to out of stock instead of deleted.`);
      await loadProducts();
    } catch (err) { alert(err.message); }
  }
});

$("product-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("product-error").textContent = "";
  try {
    const body = {
      sku: $("p-sku").value.trim(), name: $("p-name").value.trim(), category: $("p-category").value.trim(),
      color: $("p-color").value.trim(), price: Number($("p-price").value), stock: parseStock($("p-stock").value),
      description: $("p-desc").value.trim(), image_url: $("p-image").value.trim() || null,
    };
    if (editingSku) await call(`/api/seller/products/${encodeURIComponent(editingSku)}`, { method: "PUT", body });
    else await call("/api/seller/products", { method: "POST", body });
    $("product-dialog").close();
    await loadProducts();
  } catch (err) {
    $("product-error").textContent = err.message;
  }
});

$("csv").addEventListener("change", async () => {
  const file = $("csv").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  const box = $("csv-result");
  try {
    const r = await call("/api/seller/products/upload", { method: "POST", form });
    box.innerHTML = `Added ${r.created}, updated ${r.updated}.` +
      (r.errors.length ? `<br>${r.errors.length} rows skipped:<br>${r.errors.map(escapeHtml).join("<br>")}` : "");
    await loadProducts();
  } catch (err) {
    box.textContent = err.message;
  }
  box.classList.remove("hidden");
  $("csv").value = "";
});

// ── conversations ──
async function loadChats() {
  const chats = await call("/api/seller/conversations");
  $("chats-body").innerHTML = chats.length ? chats.map((c) => `
    <tr>
      <td style="white-space:nowrap">${escapeHtml(c.updated_at)}</td>
      <td>…${escapeHtml(c.customer)}</td>
      <td>${escapeHtml(c.last_message)}</td>
      <td>${c.turns}</td>
      <td><button class="btn small" data-thread="${escapeHtml(c.thread_id)}">Read</button></td>
    </tr>`).join("") : `<tr><td colspan="5" class="empty">No conversations yet. Share your store link with customers.</td></tr>`;
}

$("chats-body").addEventListener("click", async (e) => {
  const tid = e.target.dataset.thread;
  if (!tid) return;
  try {
    const h = await call(`/api/seller/conversations/${encodeURIComponent(tid)}`);
    $("transcript").innerHTML = h.messages.map((m) => `<div class="msg ${m.role}">${renderMarkdown(m.text)}</div>`).join("")
      || `<p class="muted">No messages.</p>`;
    $("chat-dialog").showModal();
  } catch (err) { alert(err.message); }
});

// ── settings ──
async function loadSettings() {
  $("s-name").value = store.name;
  $("s-fees").value = Object.entries(store.delivery_fees).map(([c, f]) => `${c}: ${f}`).join("\n");
  $("s-default").value = store.default_delivery_fee;
  $("settings-msg").textContent = "";
}

$("settings-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = $("settings-msg");
  msg.style.color = "";
  try {
    const fees = {};
    for (const line of $("s-fees").value.split("\n")) {
      if (!line.trim()) continue;
      const m = line.match(/^\s*(.+?)\s*:\s*(\d+)\s*$/);
      if (!m) throw new Error(`"${line.trim()}" should look like islamabad: 200`);
      fees[m[1]] = Number(m[2]);
    }
    store = await call("/api/seller/settings", { method: "PUT", body: {
      name: $("s-name").value.trim(), delivery_fees: fees, default_delivery_fee: Number($("s-default").value) } });
    $("store-title").textContent = store.name;
    msg.style.color = "var(--ok)";
    msg.textContent = "Saved. The assistant uses these fees from the next message.";
  } catch (err) {
    msg.textContent = err.message;
  }
});

document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => b.closest("dialog").close()));
setMode(false);
start();
