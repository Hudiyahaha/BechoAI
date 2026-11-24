// Shared helpers for the store page and the dashboard.

const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode: keep going */ } },
  remove(key) { try { localStorage.removeItem(key); } catch { /* ignore */ } },
};

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Tiny, safe Markdown: escape everything first, then allow **bold**, bullet lists and line breaks.
function renderMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  let html = "", inList = false;
  for (const raw of lines) {
    const line = raw.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    const bullet = line.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)$/);
    if (bullet) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${bullet[1]}</li>`;
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (line.trim()) html += `<p>${line}</p>`;
    }
  }
  return html + (inList ? "</ul>" : "");
}

const rs = (n) => "Rs " + Number(n).toLocaleString("en-PK");

const COLORS = {
  red: "#c8343f", maroon: "#7a1f2b", pink: "#e9a1b5", "pastel pink": "#f4c7d3", peach: "#f6c4a4", mustard: "#d6a21e",
  gold: "#c9a227", yellow: "#f1d04b", orange: "#e7863a", green: "#3f8f5a", "emerald green": "#1f7a5a", mint: "#a8dcc1",
  blue: "#3866b8", navy: "#22325e", teal: "#2a8c8c", purple: "#7d4fa3", lilac: "#c6a9e0", black: "#24201e",
  white: "#f2eee8", cream: "#efe4cc", beige: "#dccbb0", grey: "#9a9590", gray: "#9a9590", brown: "#7a5236",
};
function colorFor(name) {
  const key = String(name || "").toLowerCase();
  if (COLORS[key]) return COLORS[key];
  const word = Object.keys(COLORS).find((c) => key.includes(c));
  return word ? COLORS[word] : "#c9b8a6";
}
const ICONS = { kurta: "👘", formal: "✨", unstitched: "🧵", dupatta: "🧣", trouser: "👖" };
const iconFor = (category) => ICONS[String(category || "").toLowerCase()] || "🛍️";

function sizesText(sizes) {
  const keys = Object.keys(sizes || {});
  if (keys.length === 1 && keys[0] === "ONE SIZE") return ["One size"];
  return keys;
}

async function api(path, { method = "GET", token, body, form } = {}) {
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch((window.BECHOAI_API || "") + path, { method, headers, body: form ?? (body !== undefined ? JSON.stringify(body) : undefined) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    const err = new Error(detail || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

function swatchHtml(p, big = false) {
  const color = colorFor(p.color);
  const inner = p.image_url ? `<img src="${escapeHtml(p.image_url)}" alt="${escapeHtml(p.name)}" loading="lazy">`
                            : `<span aria-hidden="true">${iconFor(p.category)}</span>`;
  return `<div class="swatch" style="background:linear-gradient(135deg, ${color}, color-mix(in srgb, ${color} 55%, white))${big ? ";aspect-ratio:16/9" : ""}">
    ${inner}<span class="sku">${escapeHtml(p.sku)}</span></div>`;
}
