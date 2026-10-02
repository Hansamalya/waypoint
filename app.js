/* =========================================================================
   SkillGraph — Knowledge Graph Embedding on Skill and Career Recommendation
   Front end for the results of pipeline/ (O*NET + LinkedIn + Coursera KG,
   Node2Vec + HGT). All data comes from data/kg_web.json.
   ========================================================================= */
(() => {
"use strict";
const BASE = document.body.dataset.base || "";
const PAGE = document.body.dataset.page || "";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = n => Number(n || 0).toLocaleString();
const f3 = v => Number(v).toFixed(3);
const sigmoid = x => 1 / (1 + Math.exp(-x));
const DOMAIN_COL = { "AI & Machine Learning": "#2E9E4F", "Data Analytics & BI": "#1F77B4", "Cloud & DevOps": "#8C564B",
                     "Software & Web": "#9467BD", "Cybersecurity": "#D62728", "UI/UX & Design": "#C2549D" };
const MODELS = { hgt: "HGT", node2vec: "Node2Vec", overlap: "Skill overlap" };

/* ---------------------------------------------------------------- storage */
const LS = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } },
};
const Store = {
  session() { return LS.get("sg_session", null); },
  signIn(u) { LS.set("sg_session", { name: u.name, email: u.email, guest: !!u.guest }); },
  signOut() { LS.del("sg_session"); location.href = BASE + "login.html"; },
  key(k) { const s = this.session(); return `sg_${k}_${s ? s.email : "guest"}`; },
  profile() { return Object.assign({ name: this.session()?.name || "", skills: [], model: "hgt", resume: null }, LS.get(this.key("profile"), {})); },
  saveProfile(p) { LS.set(this.key("profile"), p); },
  progress(id) { return LS.get(this.key("progress_" + id), {}); },
  saveProgress(id, v) { LS.set(this.key("progress_" + id), v); },
};

/* ---------------------------------------------------------------- data */
const D = { db: null };
async function load() {
  if (D.db) return D.db;
  const r = await fetch(BASE + "data/kg_web.json");
  if (!r.ok) throw new Error("kg_web.json " + r.status);
  const db = await r.json(); D.db = db;
  db.skillIndex = Object.fromEntries(db.skills.map((s, i) => [s.name.toLowerCase(), i]));
  db.occIndex = Object.fromEntries(db.occupations.map((o, i) => [o.id, i]));
  db.teachers = db.skills.map(() => []);                        // skill → courses that teach it
  db.courses.forEach((c, ci) => c.teaches.forEach(s => db.teachers[s].push(ci)));
  db.rel = db.skills.map(() => []);                             // skill ↔ related skills
  db.related.forEach(([a, b, v]) => { db.rel[a].push([b, v]); db.rel[b].push([a, v]); });
  db.reqMap = db.occupations.map(o => new Map(o.requires.map(([s, w, d, c, src]) => [s, { s, w, d, c, src }])));
  db.occCount = db.skills.map(() => 0);
  db.occupations.forEach(o => o.requires.forEach(([s, w]) => { if (w >= 0.3) db.occCount[s]++; }));
  return db;
}

/* ---------------------------------------------------------------- skills & résumé parsing (same lexicon as kg_build.py) */
const norm = t => " " + String(t).toLowerCase().replace(/[^a-z0-9+#./]+/g, " ") + " ";
const ALIAS = { "js": "JavaScript", "reactjs": "React", "react.js": "React", "nodejs": "Node.js", "postgres": "PostgreSQL", "ms excel": "Excel",
  "powerbi": "Power BI", "ml": "Machine Learning", "amazon web services": "AWS", "gcp": "Google Cloud", "k8s": "Kubernetes", "golang": "Go",
  "ppt": "PowerPoint", "dl": "Deep Learning", "nlp": "NLP", "cv": "Computer Vision", "ux": "UX Research", "ci cd": "CI/CD" };
function resolveSkill(name) {
  const db = D.db; const k = String(name).trim().toLowerCase();
  if (k in db.skillIndex) return db.skillIndex[k];
  const a = ALIAS[k]; if (a && a.toLowerCase() in db.skillIndex) return db.skillIndex[a.toLowerCase()];
  const hit = db.skills.findIndex(s => s.name.toLowerCase().includes(k) && k.length >= 3);
  return hit >= 0 ? hit : -1;
}
function extractSkills(text) {
  const db = D.db; const nt = norm(text); const found = new Set();
  const conc = Object.fromEntries(Object.entries(db.lexicon.concepts).map(([c, p]) => [c, new RegExp(p, "i")]));
  const amb = Object.fromEntries(Object.entries(db.lexicon.ambiguous).map(([c, p]) => [c, new RegExp(p)]));
  db.skills.forEach((s, i) => {
    if (s.kind === "concept") { if (conc[s.name]?.test(nt)) found.add(i); }
    else if (amb[s.name]) { if (amb[s.name].test(text)) found.add(i); }
    else if (nt.includes(norm(s.name))) found.add(i);
  });
  for (const [k, v] of Object.entries(ALIAS)) if (nt.includes(" " + k + " ")) { const i = resolveSkill(v); if (i >= 0) found.add(i); }
  return [...found];
}
const profileSkillIdx = p => [...new Set(p.skills.map(resolveSkill).filter(i => i >= 0))];

/* ---------------------------------------------------------------- recommendation engine (mirrors kg_models.py scorers) */
function scoreAll(U, model) {
  const db = D.db;
  return db.occupations.map((o, oi) => {
    const req = db.reqMap[oi]; let tot = 0, have = 0;
    req.forEach(e => { tot += e.w; if (U.includes(e.s)) have += e.w; });
    const coverage = tot ? have / tot : 0;
    const hgt = U.length ? U.reduce((a, s) => a + sigmoid(db.scores.hgt[oi][s]), 0) / U.length : 0;
    const n2v = U.length ? U.reduce((a, s) => a + db.scores.node2vec[oi][s], 0) / U.length : 0;
    const score = model === "hgt" ? hgt : model === "node2vec" ? n2v : coverage;
    return { o, oi, score, hgt, n2v, coverage };
  }).sort((a, b) => b.score - a.score);
}
function explain(oi, U) {
  const db = D.db; const o = db.occupations[oi]; const Us = new Set(U);
  const req = [...db.reqMap[oi].values()].sort((a, b) => b.w - a.w);
  const matched = req.filter(e => Us.has(e.s)), missing = req.filter(e => !Us.has(e.s));
  const paths = [];
  for (const m of missing.slice(0, 12)) {                        // RELATED_TO bridge from a skill you already have
    const br = db.rel[m.s].filter(([x]) => Us.has(x)).sort((a, b) => b[1] - a[1])[0];
    if (br) paths.push({ kind: "related", text: `${db.skills[br[0]].name} —RELATED_TO→ ${db.skills[m.s].name} ←REQUIRES— ${o.name}`, skill: m.s });
    if (paths.length >= 2) break;
  }
  const courses = [], seen = new Set();
  for (const m of missing.slice(0, 10)) {
    for (const ci of db.teachers[m.s]) if (!seen.has(ci)) { seen.add(ci); courses.push({ ci, skill: m.s, how: "TEACHES" }); }
    if (courses.length >= 6) break;
  }
  if (courses.length < 3) for (const m of missing.slice(0, 5)) for (const ci of db.hgt_courses[m.s] || []) {
    if (!seen.has(ci) && courses.length < 5) { seen.add(ci); courses.push({ ci, skill: m.s, how: "HGT" }); }
  }
  courses.slice(0, 2).filter(c => c.how === "TEACHES").forEach(c =>
    paths.push({ kind: "course", text: `${db.courses[c.ci].name} —TEACHES→ ${db.skills[c.skill].name} ←REQUIRES— ${o.name}` }));
  return { matched, missing, courses, paths };
}

/* ---------------------------------------------------------------- UI helpers */
function photo(o, cls = "") {
  return `<div class="photo ${cls}" data-initial="${esc(o.name[0])}" style="--dc:${DOMAIN_COL[o.domain]}"><img src="${esc(o.image)}" alt="" loading="lazy" referrerpolicy="no-referrer"></div>`;
}
document.addEventListener("error", e => { const t = e.target; if (t.tagName === "IMG" && t.parentElement?.classList.contains("photo")) t.parentElement.classList.add("fallback"); }, true);
const dchip = d => `<span class="dchip" style="--dc:${DOMAIN_COL[d]}">${esc(d)}</span>`;
function toast(m) { let t = $(".toast"); if (!t) { t = document.createElement("div"); t.className = "toast"; t.setAttribute("role", "status"); document.body.appendChild(t); } t.textContent = m; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 2400); }
const loadScript = src => new Promise((res, rej) => { if ($(`script[src="${src}"]`)) return res(); const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = () => rej(new Error("Could not load " + src)); document.head.appendChild(s); });
const LOGO = `<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="15" fill="#1b2a41"/><path d="M10 21 L16 10 L22 21 Z" fill="none" stroke="#f2a541" stroke-width="1.6"/><circle cx="16" cy="10" r="3" fill="#f2a541"/><circle cx="10" cy="21" r="3" fill="#7fd1c7"/><circle cx="22" cy="21" r="3" fill="#ef8a97"/></svg>`;
function renderNav() {
  const host = $("#nav"); if (!host) return; const s = Store.session();
  const links = [["dashboard.html", "Dashboard"], ["recommendation.html", "Careers"], ["roadmap.html", "Roadmap"], ["graphs/knowledge_graph.html", "Knowledge graph"], ["results.html", "Results"]];
  const here = location.pathname.split("/").slice(-2).join("/");
  host.innerHTML = `<nav class="nav" aria-label="Main"><div class="wrap"><a class="brand" href="${BASE}index.html">${LOGO}SkillGraph</a>
    <button class="nav-toggle" aria-label="Open menu" aria-expanded="false"><span></span><span></span><span></span></button>
    <div class="nav-links" id="navLinks">${links.map(([h, t]) => `<a href="${BASE}${h}" ${here.endsWith(h) ? 'aria-current="page"' : ""}>${t}</a>`).join("")}</div>
    <div class="nav-user">${s ? `<span class="who small muted">${esc(s.name)}</span><div class="avatar" aria-hidden="true">${esc((s.name || "?")[0].toUpperCase())}</div><button class="btn btn-ghost btn-sm" id="signOut">Sign out</button>` : `<a class="btn btn-sm" href="${BASE}login.html">Sign in</a>`}</div></div></nav>`;
  $("#signOut")?.addEventListener("click", () => Store.signOut());
  const tg = $(".nav-toggle", host); tg.addEventListener("click", () => { const o = $("#navLinks").classList.toggle("open"); tg.setAttribute("aria-expanded", o); });
}
function requireSession() { if (!Store.session()) { location.replace(BASE + "login.html?next=" + encodeURIComponent(location.pathname.split("/").pop())); return false; } return true; }
function dataError(host, err) {
  console.error(err);
  host.innerHTML = `<div class="wrap"><div class="notice"><strong>The knowledge-graph data could not be loaded.</strong><br>Browsers block data files opened straight from disk.
    In VS Code press <code>F5</code> (“▶ Run SkillGraph”), or run <code>python backend/app.py</code> and open <code>http://127.0.0.1:5000</code>.</div></div>`;
}
const modelSwitch = (cur, id = "modelSel") => `<div class="seg" role="radiogroup" aria-label="Ranking model" id="${id}">${Object.entries(MODELS).map(([k, v]) =>
  `<button role="radio" aria-checked="${k === cur}" data-model="${k}">${v}</button>`).join("")}</div>`;

const Pages = {};

/* ================================================================ landing */
Pages.index = async () => {
  try {
    const db = await load(); const s = db.stats; const S = db.summary;
    const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
    set("#stNodes", num(s.nodes.total)); set("#stEdges", num(s.edges.total)); set("#stPost", num(s.postings.matched));
    set("#stAuc", f3(S["LP/REQUIRES/HGT"].AUC.mean));
    const pick = ["15-2051.00", "15-1255.00", "15-1212.00", "15-1252.00", "15-2051.01", "15-1241.00", "15-1299.07", "15-1221.00"];
    const shape = ["tall", "wide", "", "", "", "wide", "", ""];
    $("#mosaic").innerHTML = pick.map((id, i) => { const o = db.occupations[db.occIndex[id]]; if (!o) return "";
      return `<a class="tile ${shape[i]}" href="recommendation.html?occ=${o.id}">${photo(o)}<div class="tile-body"><h3>${esc(o.name)}</h3><span>${esc(o.domain)} · ${o.requires.length} required skills</span></div></a>`; }).join("");
  } catch (e) { console.warn("Open through the local server to load live numbers.", e); }
  $$("[data-start]").forEach(a => a.href = Store.session() ? "dashboard.html" : "login.html");
};

/* ================================================================ login */
async function sha(s) { if (crypto?.subtle) { const b = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)); return [...new Uint8Array(b)].map(x => x.toString(16).padStart(2, "0")).join(""); } return btoa(unescape(encodeURIComponent(s))); }
Pages.login = () => {
  const next = new URLSearchParams(location.search).get("next") || "dashboard.html";
  const go = () => location.href = /^[\w-]+\.html$/.test(next) ? next : "dashboard.html";
  let mode = "signin";
  const setMode = m => { mode = m; $$(".tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.mode === m)); $("#nameField").hidden = m === "signin";
    $("#submitBtn").textContent = m === "signin" ? "Sign in" : "Create account"; $("#authTitle").textContent = m === "signin" ? "Welcome back" : "Create your account"; $("#err").textContent = ""; };
  $$(".tabs button").forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("#authForm").addEventListener("submit", async e => {
    e.preventDefault(); const email = $("#email").value.trim().toLowerCase(), pw = $("#password").value, name = $("#name").value.trim(), err = $("#err");
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) return (err.textContent = "Enter a valid email address.");
    if (pw.length < 6) return (err.textContent = "Passwords need at least 6 characters.");
    const users = LS.get("sg_users", {}); const h = await sha(email + ":" + pw);
    if (mode === "signup") { if (!name) return (err.textContent = "Add your name."); if (users[email]) return (err.textContent = "That email already has an account. Sign in instead.");
      users[email] = { name, email, h }; LS.set("sg_users", users); Store.signIn(users[email]); go(); }
    else { if (!users[email] || users[email].h !== h) return (err.textContent = "Email or password doesn't match."); Store.signIn(users[email]); go(); }
  });
  $("#guestBtn").addEventListener("click", () => { Store.signIn({ name: "Guest", email: "guest", guest: true }); go(); });
  setMode("signin");
};

/* ================================================================ dashboard */
Pages.dashboard = async () => {
  if (!requireSession()) return; renderNav();
  let db; try { db = await load(); } catch (e) { return dataError($("#main"), e); }
  const profile = Store.profile();
  $("#pName").value = profile.name || "";
  $("#modelHost").innerHTML = modelSwitch(profile.model);
  $("#modelHost").addEventListener("click", e => { const b = e.target.closest("[data-model]"); if (!b) return; profile.model = b.dataset.model;
    $$("#modelSel button").forEach(x => x.setAttribute("aria-checked", x === b)); save(); });
  const vocab = db.skills.map((s, i) => ({ n: s.name, t: s.kind === "concept" ? "Concept" : "Tool", c: db.occCount[i] }));

  const renderChips = () => {
    const U = new Set(profileSkillIdx(profile));
    $("#skillChips").innerHTML = profile.skills.length ? profile.skills.map(s => `<span class="chip ${resolveSkill(s) >= 0 ? "have" : ""}" title="${resolveSkill(s) >= 0 ? "In the knowledge graph" : "Not in the knowledge graph — ignored by the models"}">${esc(s)}<button type="button" aria-label="Remove ${esc(s)}" data-rm="${esc(s)}">×</button></span>`).join("")
      : `<span class="muted small">No skills yet. Add a few, or upload your résumé.</span>`;
    $("#skillCount").textContent = `${U.size} in graph`;
  };
  const addSkill = s => { if (s && !profile.skills.some(x => x.toLowerCase() === s.toLowerCase())) profile.skills.push(s); };
  $("#skillChips").addEventListener("click", e => { const b = e.target.closest("[data-rm]"); if (b) { profile.skills = profile.skills.filter(x => x !== b.dataset.rm); save(); } });
  const inp = $("#skillInput"), box = $("#suggest"); let items = [], active = -1;
  const close = () => { box.hidden = true; active = -1; };
  inp.addEventListener("input", () => { const q = inp.value.trim().toLowerCase(); if (!q) return close();
    items = vocab.filter(v => v.n.toLowerCase().includes(q) && !profile.skills.includes(v.n)).sort((a, b) => (a.n.toLowerCase().indexOf(q) - b.n.toLowerCase().indexOf(q)) || b.c - a.c).slice(0, 8);
    box.innerHTML = items.length ? items.map((v, i) => `<button type="button" data-i="${i}">${esc(v.n)}<em>${v.t} · ${v.c} occupations</em></button>`).join("") : `<div class="small muted" style="padding:10px 14px">No skill called “${esc(inp.value)}” in the graph.</div>`;
    box.hidden = false; });
  inp.addEventListener("keydown", e => { const b = $$("button", box);
    if (e.key === "ArrowDown") { e.preventDefault(); active = Math.min(active + 1, b.length - 1); } else if (e.key === "ArrowUp") { e.preventDefault(); active = Math.max(active - 1, 0); }
    else if (e.key === "Enter") { e.preventDefault(); (b[active] || b[0])?.click(); return; } else if (e.key === "Escape") return close();
    b.forEach((x, i) => x.classList.toggle("active", i === active)); });
  box.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; addSkill(items[+b.dataset.i].n); inp.value = ""; close(); save(); });
  document.addEventListener("click", e => { if (!e.target.closest(".autocomplete")) close(); });

  // résumé
  const dz = $("#dropzone"), file = $("#resumeFile");
  dz.addEventListener("click", () => file.click());
  dz.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  ["dragenter", "dragover"].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", e => e.dataTransfer.files[0] && handle(e.dataTransfer.files[0]));
  file.addEventListener("change", () => file.files[0] && handle(file.files[0]));
  async function read(f) {
    const ext = f.name.split(".").pop().toLowerCase();
    try {
      if (ext === "pdf") { await loadScript(BASE + "js/vendor/pdf.min.js"); pdfjsLib.GlobalWorkerOptions.workerSrc = BASE + "js/vendor/pdf.worker.min.js";
        const pdf = await pdfjsLib.getDocument({ data: await f.arrayBuffer() }).promise; let t = "";
        for (let i = 1; i <= pdf.numPages; i++) t += (await (await pdf.getPage(i)).getTextContent()).items.map(x => x.str + (x.hasEOL ? "\n" : " ")).join("") + "\n";
        return t; }
      if (ext === "docx") { await loadScript(BASE + "js/vendor/mammoth.browser.min.js"); return (await mammoth.extractRawText({ arrayBuffer: await f.arrayBuffer() })).value; }
      if (["txt", "md"].includes(ext)) return f.text();
    } catch (err) { const fd = new FormData(); fd.append("file", f); const r = await fetch(BASE + "api/resume", { method: "POST", body: fd }).catch(() => null); if (r?.ok) return (await r.json()).text; throw err; }
    throw new Error("Upload a PDF, DOCX or TXT résumé.");
  }
  async function handle(f) {
    const st = $("#resumeStatus"); st.innerHTML = `<p class="muted">Reading ${esc(f.name)}…</p>`;
    try {
      const idx = extractSkills(await read(f)); const names = idx.map(i => db.skills[i].name);
      const fresh = names.filter(n => !profile.skills.some(x => x.toLowerCase() === n.toLowerCase()));
      profile.resume = { name: f.name, found: names.length };
      st.innerHTML = `<div class="stack" style="gap:12px;margin-top:16px"><p><strong>${names.length} graph skills found</strong> in ${esc(f.name)}</p>
        <div class="chips">${names.map(n => `<span class="chip ${fresh.includes(n) ? "route" : "have"}">${esc(n)}</span>`).join("") || '<span class="muted">No skills from the graph matched. Add them manually.</span>'}</div>
        ${fresh.length ? `<div><button class="btn btn-route btn-sm" id="addFound">Add ${fresh.length} skill${fresh.length > 1 ? "s" : ""} to my profile</button></div>` : ""}</div>`;
      $("#addFound")?.addEventListener("click", () => { fresh.forEach(addSkill); save(); toast(`Added ${fresh.length} skills from your résumé`); $("#addFound").remove(); });
      save(true);
    } catch (err) { st.innerHTML = `<p class="form-error">Couldn't read that file. ${esc(err.message)}</p>`; }
  }

  let charts = {};
  function save(quiet) { profile.name = $("#pName").value.trim(); Store.saveProfile(profile); renderChips(); insights(); if (!quiet) $("#savedAt").textContent = "Saved"; }
  $("#pName").addEventListener("input", () => { clearTimeout(save._t); save._t = setTimeout(save, 300); });

  function insights() {
    const U = profileSkillIdx(profile);
    $("#greet").textContent = profile.name ? `Hi ${profile.name.split(" ")[0]}, here's where your skills lead` : "Your skill dashboard";
    if (!U.length) $("#topMatches").innerHTML = `<div class="empty"><h3>Add skills to see matches</h3><p>Upload a résumé or type a few skills on the left.</p></div>`;
    else $("#topMatches").innerHTML = scoreAll(U, profile.model).slice(0, 4).map(r => `<a class="match-mini" href="recommendation.html?occ=${r.o.id}">${photo(r.o)}
      <div><h4>${esc(r.o.name)}</h4><p class="small muted">${esc(r.o.domain)} · covers ${Math.round(r.coverage * 100)}% of weighted skills</p>
      <div class="bar"><i style="width:${Math.round(r.coverage * 100)}%"></i></div></div><div class="score" title="${MODELS[profile.model]} score">${profile.model === "overlap" ? Math.round(r.score * 100) + "<small>%</small>" : r.score.toFixed(2)}</div></a>`).join("");
    $("#modelNote").textContent = `Ranked by ${MODELS[profile.model]}. ${profile.model === "hgt" ? "Mean HGT link score between the occupation and your skills." : profile.model === "node2vec" ? "Mean Node2Vec cosine similarity." : "Share of the occupation's weighted skills you have."}`;
    const st = db.stats;
    $("#kpis").innerHTML = [[num(st.nodes.total), "Graph nodes"], [num(st.edges.total), "Graph edges"], [num(st.postings.matched), "Job postings matched"], [st.avg_degree, "Average degree"]]
      .map(([v, l]) => `<div class="kpi"><strong>${v}</strong><span>${l}</span></div>`).join("");
    draw(U);
  }
  function draw(U) {
    if (!window.Chart) return; Chart.defaults.font.family = getComputedStyle(document.body).fontFamily; Chart.defaults.color = "#5b7083";
    const top = db.skills.map((s, i) => ({ n: s.name, i, c: db.occCount[i] })).sort((a, b) => b.c - a.c).slice(0, 14);
    const Us = new Set(U);
    const domains = Object.keys(DOMAIN_COL);
    const domCov = domains.map(d => { const os = db.occupations.map((o, oi) => [o, oi]).filter(([o]) => o.domain === d);
      return os.length ? os.reduce((a, [, oi]) => { let t = 0, h = 0; db.reqMap[oi].forEach(e => { t += e.w; if (Us.has(e.s)) h += e.w; }); return a + (t ? h / t : 0); }, 0) / os.length * 100 : 0; });
    const src = db.stats.requires_sources;
    const cfg = {
      skills: { type: "bar", data: { labels: top.map(t => t.n), datasets: [{ data: top.map(t => t.c), backgroundColor: top.map(t => Us.has(t.i) ? "#1f7a74" : "#c9d5d0"), borderRadius: 6 }] },
        options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `${c.raw} occupations require it (weight ≥ 0.3)` } } }, scales: { x: { grid: { color: "#eef2f0" } }, y: { grid: { display: false } } } } },
      domain: { type: "bar", data: { labels: domains, datasets: [{ data: domCov, backgroundColor: domains.map(d => DOMAIN_COL[d]), borderRadius: 6 }] },
        options: { maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `${c.raw.toFixed(1)}% of weighted skills covered on average` } } }, scales: { y: { max: 100, ticks: { callback: v => v + "%" } }, x: { grid: { display: false }, ticks: { font: { size: 10 } } } } } },
      source: { type: "doughnut", data: { labels: ["O*NET + postings", "O*NET only", "Postings only"], datasets: [{ data: [src.both, src.onet, src.postings], backgroundColor: ["#1b2a41", "#6c8ead", "#f2a541"], borderWidth: 0 }] },
        options: { maintainAspectRatio: false, cutout: "62%", plugins: { legend: { position: "right", labels: { boxWidth: 10, boxHeight: 10 } } } } },
    };
    for (const [k, c] of Object.entries(cfg)) { charts[k]?.destroy(); charts[k] = new Chart($("#ch-" + k), c); }
  }
  renderChips(); insights();
};

/* ================================================================ recommendations */
function gapRows(list, Us) {
  return list.map(e => `<div class="gap-row ${Us.has(e.s) ? "have" : "gap"}"><span class="name">${esc(D.db.skills[e.s].name)}</span>
    <span class="track" role="img" aria-label="weight ${f3(e.w)}"><i style="width:${Math.round(e.w * 100)}%"></i></span>
    <span class="val" title="demand ${Math.round(e.d * 100)}% of postings · confidence ${e.c.toFixed(2)} · source ${e.src}">${Math.round(e.d * 100)}% · c ${e.c.toFixed(2)}</span></div>`).join("");
}
Pages.recommendation = async () => {
  if (!requireSession()) return; renderNav();
  let db; try { db = await load(); } catch (e) { return dataError($("#main"), e); }
  const profile = Store.profile(); const U = profileSkillIdx(profile); const Us = new Set(U);
  let model = profile.model, dom = "All";
  let sel = new URLSearchParams(location.search).get("occ");
  if (!U.length) $("#hint").innerHTML = `<div class="notice">Your profile has no skills from the graph yet, so every occupation scores the same. <a href="dashboard.html">Add skills or upload a résumé</a>.</div>`;
  $("#controls").innerHTML = ["All", ...Object.keys(DOMAIN_COL)].map(d => `<button class="pill" aria-pressed="${d === dom}" data-dom="${esc(d)}">${esc(d)}</button>`).join("") + `<span style="margin-left:auto"></span>` + modelSwitch(model);
  $("#controls").addEventListener("click", e => {
    const d = e.target.closest("[data-dom]"); const m = e.target.closest("[data-model]");
    if (d) { dom = d.dataset.dom; $$("#controls .pill").forEach(x => x.setAttribute("aria-pressed", x === d)); }
    if (m) { model = m.dataset.model; profile.model = model; Store.saveProfile(profile); $$("#modelSel button").forEach(x => x.setAttribute("aria-checked", x === m)); }
    if (d || m) { list(); detail(); }
  });
  let ranked = [];
  function list() {
    ranked = scoreAll(U, model); if (!sel) sel = ranked[0].o.id;
    $("#recList").innerHTML = ranked.filter(r => dom === "All" || r.o.domain === dom).map((r, i) => `<button class="rec-card" role="option" aria-selected="${r.o.id === sel}" data-id="${r.o.id}">
      ${photo(r.o)}<div><div style="display:flex;justify-content:space-between;gap:8px"><h3>${esc(r.o.name)}</h3><span class="score" style="font-size:1.1rem">${model === "overlap" ? Math.round(r.score * 100) + "%" : r.score.toFixed(2)}</span></div>
      <div class="bar"><i style="width:${Math.round(r.coverage * 100)}%;background:${DOMAIN_COL[r.o.domain]}"></i></div>
      <div class="rec-meta"><span>${esc(r.o.domain)}</span><span>${Math.round(r.coverage * 100)}% covered</span></div></div></button>`).join("");
  }
  $("#recList").addEventListener("click", e => { const b = e.target.closest(".rec-card"); if (!b) return; sel = b.dataset.id; history.replaceState(null, "", "?occ=" + sel); list(); detail(); if (innerWidth < 1080) $("#detail").scrollIntoView({ behavior: "smooth" }); });
  function detail() {
    const oi = db.occIndex[sel]; const o = db.occupations[oi]; const r = ranked.find(x => x.oi === oi); const rank = ranked.indexOf(r) + 1;
    const ex = explain(oi, U); const req = [...db.reqMap[oi].values()].sort((a, b) => b.w - a.w).slice(0, 12);
    $("#detail").innerHTML = `
      <div class="detail-hero">${photo(o)}<div><div>${dchip(o.domain)}<h2 style="margin-top:10px">${esc(o.name)}</h2><p>O*NET ${esc(o.id)} · ${esc(o.onet_title)}</p></div>
        <div class="big-score">#${rank}<small>of 27 by ${MODELS[model]}</small></div></div></div>
      <div class="detail-body">
        <p>${esc(o.description)}</p>
        <div class="breakdown"><div><strong>${r.hgt.toFixed(3)}</strong><span>HGT score</span></div><div><strong>${r.n2v.toFixed(3)}</strong><span>Node2Vec score</span></div><div><strong>${Math.round(r.coverage * 100)}%</strong><span>Weighted skill coverage</span></div></div>
        <section><div class="panel-head"><h3>Skill gap</h3><div class="legend"><span class="l-have">You have</span><span class="l-gap">To learn</span></div></div>
          <p class="small muted" style="margin:-8px 0 12px">Bar = edge weight (recency-weighted demand × confidence). Right: share of ${num(o.postings)} postings that mention it · confidence c.</p>${gapRows(req, Us)}</section>
        <section><h3 style="margin-bottom:10px">Why this recommendation</h3>
          <p style="margin-bottom:8px"><strong>Matched (${ex.matched.length}):</strong> ${ex.matched.slice(0, 8).map(e => `<span class="chip have">${esc(db.skills[e.s].name)} · ${e.w.toFixed(2)}</span>`).join(" ") || '<span class="muted">none yet</span>'}</p>
          <p style="margin-bottom:12px"><strong>Top missing:</strong> ${ex.missing.slice(0, 6).map(e => `<span class="chip gap">${esc(db.skills[e.s].name)} · ${e.w.toFixed(2)}</span>`).join(" ")}</p>
          ${ex.paths.map(p => `<div class="kgpath ${p.kind}">${esc(p.text)}</div>`).join("") || '<p class="small muted">Add skills to see graph paths.</p>'}</section>
        <section><div class="panel-head"><h3>Courses for your missing skills</h3><span class="small muted">Coursera</span></div>
          ${ex.courses.map(c => { const k = db.courses[c.ci]; return `<a class="course" href="${esc(k.url)}" target="_blank" rel="noopener"><div><h4>${esc(k.name)}</h4>
            <div class="meta">${esc(k.org)} · ${esc(k.level)} · teaches <span class="chip gap" style="padding:1px 8px">${esc(db.skills[c.skill].name)}</span>${c.how === "HGT" ? ' · <em>suggested by HGT</em>' : ""}</div></div>${k.rating ? `<span class="stars">★ ${k.rating.toFixed(1)}</span>` : ""}</a>`; }).join("") || '<p class="muted">No course in the catalogue teaches these skills.</p>'}</section>
        ${o.tasks.length ? `<section><h3 style="margin-bottom:8px">What the work involves (O*NET)</h3><ul class="tasks">${o.tasks.map(t => `<li>${esc(t)}</li>`).join("")}</ul></section>` : ""}
        <div style="display:flex;gap:10px;flex-wrap:wrap"><a class="btn btn-route" href="roadmap.html?occ=${o.id}">Build my roadmap</a><a class="btn btn-ghost" href="graphs/knowledge_graph.html?focus=${o.id}">Show in the graph</a></div>
      </div>`;
  }
  list(); detail();
};

/* ================================================================ roadmap */
Pages.roadmap = async () => {
  if (!requireSession()) return; renderNav();
  let db; try { db = await load(); } catch (e) { return dataError($("#main"), e); }
  const profile = Store.profile();
  const ranked = scoreAll(profileSkillIdx(profile), profile.model);
  let id = new URLSearchParams(location.search).get("occ") || ranked[0].o.id;
  $("#occPick").innerHTML = `<optgroup label="Your top matches">${ranked.slice(0, 5).map(r => `<option value="${r.o.id}">${esc(r.o.name)}</option>`).join("")}</optgroup>
    <optgroup label="All occupations">${[...db.occupations].sort((a, b) => a.name.localeCompare(b.name)).map(o => `<option value="${o.id}">${esc(o.name)}</option>`).join("")}</optgroup>`;
  $("#occPick").value = id; $("#occPick").addEventListener("change", e => { id = e.target.value; history.replaceState(null, "", "?occ=" + id); draw(); });
  function draw() {
    const U = profileSkillIdx(profile); const Us = new Set(U); const oi = db.occIndex[id]; const o = db.occupations[oi];
    const ex = explain(oi, U); const miss = ex.missing.filter(e => e.w >= 0.15);
    const quick = miss.filter(e => db.rel[e.s].some(([x]) => Us.has(x))).slice(0, 5);
    const core = miss.filter(e => !quick.includes(e)).slice(0, 6);
    const crs = [...new Set([...db.prepares[o.id], ...ex.courses.map(c => c.ci)])].slice(0, 6);
    const prog = Store.progress(id);
    const stages = [
      { t: "Quick wins", sub: "Missing skills that the graph links (RELATED_TO) to a skill you already have.", items: quick.map(e => ({ k: "s:" + e.s, label: db.skills[e.s].name, meta: `weight ${e.w.toFixed(2)} · via ${db.skills[db.rel[e.s].find(([x]) => Us.has(x))[0]].name}` })) },
      { t: "Core gaps", sub: "The most heavily weighted skills you are still missing.", items: core.map(e => ({ k: "s:" + e.s, label: db.skills[e.s].name, meta: `weight ${e.w.toFixed(2)} · in ${Math.round(e.d * 100)}% of postings` })) },
      { t: "Courses", sub: "Courses linked to this occupation (PREPARES_FOR) or to your missing skills (TEACHES).", items: crs.map(ci => ({ k: "c:" + ci, label: db.courses[ci].name, meta: `${db.courses[ci].org} · ${db.courses[ci].level}`, url: db.courses[ci].url })) },
      { t: "Show the work", sub: "Portfolio projects based on the occupation's O*NET tasks.", items: o.tasks.slice(0, 3).map((t, i) => ({ k: "p:" + i, label: t })) },
    ];
    let done = 0, total = 0;
    $("#timeline").innerHTML = stages.map((s, i) => `<article class="stage panel"><div class="stage-marker">${i + 1}</div><div class="stage-head"><h3>${s.t}</h3></div>
      <p class="muted" style="margin-bottom:12px">${s.sub}</p>${s.items.length ? s.items.map(it => { total++; if (prog[it.k]) done++;
        return `<label class="check"><input type="checkbox" data-key="${esc(it.k)}" ${prog[it.k] ? "checked" : ""}><span>${it.url ? `<a href="${esc(it.url)}" target="_blank" rel="noopener">${esc(it.label)}</a>` : esc(it.label)}${it.meta ? ` <span class="small muted">· ${esc(it.meta)}</span>` : ""}</span></label>`; }).join("") : '<p class="small muted">Nothing here — you already cover this stage.</p>'}</article>`).join("");
    const pct = total ? Math.round(100 * done / total) : 0;
    $("#roadHero").innerHTML = `${photo(o)}<div>${dchip(o.domain)}<h1 style="font-size:clamp(2.1rem,5vw,3.6rem)">Your route to ${esc(o.name)}</h1>
      <p>You cover ${Math.round(scoreAll(U, "overlap").find(r => r.oi === oi).coverage * 100)}% of this occupation's weighted skills. ${miss.length} weighted skills are still missing; the plan starts with the ones closest to what you know.</p>
      <div class="road-stats"><div><strong>${ex.matched.length}</strong><span>skills matched</span></div><div><strong>${quick.length + core.length}</strong><span>skills in the plan</span></div><div><strong>${crs.length}</strong><span>courses</span></div></div>
      <div class="progress-line"><i style="width:${pct}%"></i></div><span class="small" style="color:#c3d0dc">${done} of ${total} steps done</span></div>`;
    $("#timeline").onchange = e => { const k = e.target.dataset.key; if (!k) return; const p = Store.progress(id); p[k] = e.target.checked; Store.saveProgress(id, p);
      if (e.target.checked && k.startsWith("s:")) { const n = db.skills[+k.slice(2)].name; if (!profile.skills.includes(n)) { profile.skills.push(n); Store.saveProfile(profile); toast(`${n} added to your skills`); } }
      draw(); };
  }
  draw();
};

/* ================================================================ results */
Pages.results = async () => {
  renderNav();
  let db; try { db = await load(); } catch (e) { return dataError($("#main"), e); }
  const S = db.summary, st = db.stats; const mv = (k, m) => S[k]?.[m] ? `${S[k][m].mean.toFixed(3)} <small class="muted">± ${S[k][m].sd.toFixed(3)}</small>` : "—";
  $("#kgStats").innerHTML = [[num(st.nodes.total), "nodes"], [num(st.edges.total), "edges"], [st.avg_degree, "average degree (2E/N)"], [st.lcc_pct + "%", "in the largest component"], [st.schema_violations, "schema violations"], [st.duplicate_edges + st.dangling_edges, "duplicate or dangling edges"]]
    .map(([v, l]) => `<div class="kpi"><strong>${v}</strong><span>${l}</span></div>`).join("");
  $("#kgTypes").innerHTML = `<table class="mtable"><thead><tr><th>Nodes</th><th>Count</th><th>Edges</th><th>Count</th></tr></thead><tbody>
    <tr><td>Occupation</td><td>${st.nodes.occupation}</td><td>REQUIRES (occupation → skill)</td><td>${num(st.edges.REQUIRES)}</td></tr>
    <tr><td>Skill</td><td>${st.nodes.skill}</td><td>TEACHES (course → skill)</td><td>${num(st.edges.TEACHES)}</td></tr>
    <tr><td>Course</td><td>${st.nodes.course}</td><td>PREPARES_FOR (course → occupation)</td><td>${num(st.edges.PREPARES_FOR)}</td></tr>
    <tr><td></td><td></td><td>RELATED_TO (skill ↔ skill)</td><td>${num(st.edges.RELATED_TO)}</td></tr></tbody></table>`;
  const lp = rel => `<table class="mtable"><thead><tr><th>Model</th><th>AUC</th><th>Hits@10</th><th>MRR</th></tr></thead><tbody>${["Popularity", "Node2Vec", "HGT"].map(m =>
    `<tr class="${m === "HGT" ? "hl" : ""}"><td>${m}</td><td>${mv(`LP/${rel}/${m}`, "AUC")}</td><td>${mv(`LP/${rel}/${m}`, "Hits@10")}</td><td>${mv(`LP/${rel}/${m}`, "MRR")}</td></tr>`).join("")}</tbody></table>`;
  $("#lpReq").innerHTML = lp("REQUIRES"); $("#lpTea").innerHTML = lp("TEACHES");
  const rec = p => `<table class="mtable"><thead><tr><th>Model</th><th>Hits@1</th><th>MRR</th><th>Recall@10</th><th>NDCG@10</th></tr></thead><tbody>${["Random", "Skill overlap", "Node2Vec", "HGT"].map(m =>
    `<tr class="${m === "HGT" ? "hl" : ""}"><td>${m}</td>${["Hits@1", "MRR", "Recall@10", "NDCG@10"].map(k => `<td>${mv(`REC/${p}/${m}`, k)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  $("#recSim").innerHTML = rec("simulated"); $("#recHeld").innerHTML = rec("heldout_postings");
  $("#course").innerHTML = `<table class="mtable"><thead><tr><th>Model</th><th>MRR</th><th>Recall@10</th><th>NDCG@10</th></tr></thead><tbody>${["Popularity", "Node2Vec", "HGT"].map(m =>
    `<tr class="${m === "HGT" ? "hl" : ""}"><td>${m}</td>${["MRR", "Recall@10", "NDCG@10"].map(k => `<td>${mv(`COURSE/${m}`, k)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  const np = db.n_profiles; $("#nSim").textContent = num(np["n_profiles/simulated"]); $("#nHeld").textContent = num(np["n_profiles/heldout_postings"]);
  $("#nTest").textContent = `${num(np.n_test.REQUIRES)} occupation→skill and ${num(np.n_test.TEACHES)} course→skill test edges`;
  if (window.Chart) {
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily; Chart.defaults.color = "#5b7083";
    const g = (k, m) => S[k][m].mean;
    new Chart($("#ch-lp"), { type: "bar", data: { labels: ["AUC occ→skill", "MRR occ→skill", "AUC course→skill", "MRR course→skill"],
      datasets: ["Popularity", "Node2Vec", "HGT"].map((m, i) => ({ label: m, backgroundColor: ["#c9d5d0", "#e2b64b", "#7b61b8"][i], borderRadius: 4,
        data: [g(`LP/REQUIRES/${m}`, "AUC"), g(`LP/REQUIRES/${m}`, "MRR"), g(`LP/TEACHES/${m}`, "AUC"), g(`LP/TEACHES/${m}`, "MRR")] })) },
      options: { maintainAspectRatio: false, scales: { y: { max: 1 } }, plugins: { legend: { position: "bottom" } } } });
    new Chart($("#ch-rec"), { type: "bar", data: { labels: ["MRR · simulated", "Recall@10 · simulated", "MRR · held-out postings", "Recall@10 · held-out postings"],
      datasets: ["Random", "Skill overlap", "Node2Vec", "HGT"].map((m, i) => ({ label: m, backgroundColor: ["#e3e8e6", "#9db3d6", "#e2b64b", "#7b61b8"][i], borderRadius: 4,
        data: [g(`REC/simulated/${m}`, "MRR"), g(`REC/simulated/${m}`, "Recall@10"), g(`REC/heldout_postings/${m}`, "MRR"), g(`REC/heldout_postings/${m}`, "Recall@10")] })) },
      options: { maintainAspectRatio: false, scales: { y: { max: 1 } }, plugins: { legend: { position: "bottom" } } } });
  }
};

/* ================================================================ knowledge graph explorer */
Pages.graph = async () => {
  if (!requireSession()) return; renderNav();
  let db; try { db = await load(); } catch (e) { return dataError($("#main"), e); }
  if (!window.d3) { $("#main").innerHTML = `<div class="wrap"><div class="notice">js/vendor/d3.min.js didn't load. Check the file is present.</div></div>`; return; }
  const profile = Store.profile(); const Us = new Set(profileSkillIdx(profile));
  const focus = new URLSearchParams(location.search).get("focus");
  const state = { minW: 0.4, courses: false, dom: "All" };
  $("#domSel").innerHTML = ["All", ...Object.keys(DOMAIN_COL)].map(d => `<option>${esc(d)}</option>`).join("");
  const svg = d3.select("#graph"), host = $("#graphHost"); const g = svg.append("g");
  svg.call(d3.zoom().scaleExtent([.3, 4]).on("zoom", e => g.attr("transform", e.transform)));
  let sim;
  function build() {
    const W = host.clientWidth, H = host.clientHeight; svg.attr("viewBox", [0, 0, W, H]); g.selectAll("*").remove(); sim?.stop();
    const occs = db.occupations.map((o, oi) => ({ o, oi })).filter(x => state.dom === "All" || x.o.domain === state.dom)
      .sort((a, b) => a.o.domain.localeCompare(b.o.domain) || a.o.name.localeCompare(b.o.name));
    const R = Math.min(W, H) * 0.42; const nodes = [], links = [], idx = new Map();
    const add = n => { if (!idx.has(n.id)) { idx.set(n.id, n); nodes.push(n); } return idx.get(n.id); };
    occs.forEach(({ o, oi }, i) => { const a = 2 * Math.PI * i / occs.length - Math.PI / 2;
      add({ id: "o" + oi, type: "occ", label: o.name, color: DOMAIN_COL[o.domain], r: 11, fx: W / 2 + R * Math.cos(a), fy: H / 2 + R * Math.sin(a), oi }); });
    occs.forEach(({ oi }) => db.reqMap[oi].forEach(e => { if (e.w >= state.minW) { add({ id: "s" + e.s, type: "skill", label: db.skills[e.s].name, s: e.s }); links.push({ source: "o" + oi, target: "s" + e.s, w: e.w, kind: "req" }); } }));
    if (state.courses) nodes.filter(n => n.type === "skill").forEach(n => db.teachers[n.s].slice(0, 2).forEach(ci => { add({ id: "c" + ci, type: "course", label: db.courses[ci].name, ci }); links.push({ source: "c" + ci, target: n.id, w: .3, kind: "tea" }); }));
    nodes.forEach(n => { n.deg = links.filter(l => l.source === n.id || l.target === n.id).length; if (n.type === "skill") n.r = 4 + Math.min(n.deg, 14) * .9; if (n.type === "course") n.r = 4; });
    const link = g.append("g").selectAll("line").data(links).join("line").attr("stroke", l => l.kind === "tea" ? "#6c8ead" : idx.get(l.source).color).attr("stroke-opacity", l => .12 + .4 * l.w).attr("stroke-width", l => .5 + 1.5 * l.w);
    const node = g.append("g").selectAll("g").data(nodes).join("g").style("cursor", "pointer")
      .call(d3.drag().on("start", (e, d) => { if (!e.active) sim.alphaTarget(.2).restart(); d.fx = d.x; d.fy = d.y; }).on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); if (d.type !== "occ") d.fx = d.fy = null; }));
    node.append("circle").attr("r", d => d.r).attr("fill", d => d.type === "occ" ? d.color : d.type === "course" ? "#6c8ead" : Us.has(d.s) ? "#1f7a74" : "#f2a541").attr("stroke", "#fff").attr("stroke-width", 1.2);
    const label = node.append("text").text(d => d.label.length > 28 ? d.label.slice(0, 26) + "…" : d.label).attr("font-size", d => d.type === "occ" ? 11.5 : 10).attr("font-weight", d => d.type === "occ" ? 700 : 400)
      .attr("fill", d => d.type === "occ" ? d.color : "#1b2a41").attr("paint-order", "stroke").attr("stroke", "#f8faf9").attr("stroke-width", 3).attr("dy", "0.35em")
      .attr("x", d => d.type === "occ" ? (d.fx < W / 2 ? -(d.r + 5) : d.r + 5) : d.r + 4).attr("text-anchor", d => d.type === "occ" && d.fx < W / 2 ? "end" : "start")
      .style("display", d => d.type === "occ" || (d.type === "skill" && d.deg >= 5) ? null : "none");
    const nb = new Map(nodes.map(n => [n.id, new Set([n.id])])); links.forEach(l => { nb.get(l.source).add(l.target); nb.get(l.target).add(l.source); });
    sim = d3.forceSimulation(nodes).force("link", d3.forceLink(links).id(d => d.id).distance(l => l.kind === "tea" ? 30 : 90).strength(l => .05 + .25 * l.w))
      .force("charge", d3.forceManyBody().strength(-45)).force("collide", d3.forceCollide(d => d.r + 3)).force("x", d3.forceX(W / 2).strength(.03)).force("y", d3.forceY(H / 2).strength(.03))
      .on("tick", () => { link.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y); node.attr("transform", d => `translate(${d.x},${d.y})`); });
    let sticky = null;
    const hl = d => { const s = d ? nb.get(d.id) : null; node.style("opacity", n => !s || s.has(n.id) ? 1 : .12); link.style("opacity", l => !s || l.source.id === d.id || l.target.id === d.id ? 1 : .04);
      label.style("display", n => (s && s.has(n.id)) || (!s && (n.type === "occ" || (n.type === "skill" && n.deg >= 5))) ? null : "none"); };
    node.on("mouseenter", (e, d) => hl(d)).on("mouseleave", () => hl(sticky)).on("click", (e, d) => { e.stopPropagation(); sticky = d; hl(d); info(d, nb); });
    svg.on("click", () => { sticky = null; hl(null); info(null); });
    $("#gSearch").oninput = e => { const q = e.target.value.trim().toLowerCase(); if (!q) return hl(sticky); const h = nodes.find(n => n.label.toLowerCase().includes(q)); if (h) hl(h); };
    if (focus && idx.get("o" + db.occIndex[focus])) { const f = idx.get("o" + db.occIndex[focus]); sticky = f; setTimeout(() => { hl(f); info(f, nb); }, 400); }
    $("#gStats").textContent = `${nodes.length} nodes · ${links.length} edges shown`;
  }
  function info(d, nb) {
    const box = $("#nodeInfo");
    if (!d) { box.innerHTML = `<h3>Explore the graph</h3><p class="small muted" style="margin-top:6px">Occupations sit on the ring, grouped by domain. Skills sit between the occupations that require them. Teal skills are on your profile. Click any node.</p>`; return; }
    if (d.type === "occ") { const o = db.occupations[d.oi]; const req = [...db.reqMap[d.oi].values()].sort((a, b) => b.w - a.w).slice(0, 10);
      box.innerHTML = `${dchip(o.domain)}<h3 style="margin-top:10px">${esc(o.name)}</h3><p class="small muted" style="margin:4px 0 10px">${o.requires.length} required skills · ${num(o.postings)} postings</p>
        <div class="chips">${req.map(e => `<span class="chip ${Us.has(e.s) ? "have" : "gap"}">${esc(db.skills[e.s].name)} · ${e.w.toFixed(2)}</span>`).join("")}</div>
        <div style="display:flex;gap:8px;margin-top:14px;flex-wrap:wrap"><a class="btn btn-sm" href="../recommendation.html?occ=${o.id}">Details</a><a class="btn btn-sm btn-route" href="../roadmap.html?occ=${o.id}">Roadmap</a></div>`; }
    else if (d.type === "skill") { const occ = db.occupations.map((o, oi) => [o, db.reqMap[oi].get(d.s)]).filter(([, e]) => e).sort((a, b) => b[1].w - a[1].w);
      const rel = db.rel[d.s].sort((a, b) => b[1] - a[1]).slice(0, 6);
      box.innerHTML = `<span class="chip ${Us.has(d.s) ? "have" : "gap"}">${Us.has(d.s) ? "On your profile" : "Not on your profile"}</span><h3 style="margin-top:10px">${esc(d.label)}</h3>
        <p class="small muted" style="margin:6px 0">Required by ${occ.length} occupations:</p><div class="chips">${occ.slice(0, 8).map(([o, e]) => `<span class="chip">${esc(o.name)} · ${e.w.toFixed(2)}</span>`).join("")}</div>
        ${rel.length ? `<p class="small muted" style="margin:10px 0 6px">RELATED_TO (co-occurs in postings):</p><div class="chips">${rel.map(([x, v]) => `<span class="chip sky">${esc(db.skills[x].name)} · ${v.toFixed(2)}</span>`).join("")}</div>` : ""}
        ${db.teachers[d.s].length ? `<p class="small muted" style="margin:10px 0 6px">Taught by:</p>${db.teachers[d.s].slice(0, 3).map(ci => `<div class="small">• ${esc(db.courses[ci].name)}</div>`).join("")}` : ""}`; }
    else { const c = db.courses[d.ci]; box.innerHTML = `<span class="chip sky">Course</span><h3 style="margin-top:10px">${esc(c.name)}</h3><p class="small muted" style="margin-top:6px">${esc(c.org)} · ${esc(c.level)}${c.rating ? " · ★ " + c.rating : ""}</p>
        <p class="small" style="margin-top:8px">Teaches: ${c.teaches.map(s => esc(db.skills[s].name)).join(", ")}</p><a class="btn btn-sm" style="margin-top:12px" href="${esc(c.url)}" target="_blank" rel="noopener">Find on Coursera</a>`; }
  }
  $("#gWeight").addEventListener("input", e => { state.minW = +e.target.value; $("#gWeightOut").textContent = (+e.target.value).toFixed(2); build(); });
  $("#gCourses").addEventListener("change", e => { state.courses = e.target.checked; build(); });
  $("#domSel").addEventListener("change", e => { state.dom = e.target.value; build(); });
  addEventListener("resize", () => { clearTimeout(build._t); build._t = setTimeout(build, 250); });
  build(); info(null);
};

const boot = { index: Pages.index, login: Pages.login, dashboard: Pages.dashboard, recommendation: Pages.recommendation, roadmap: Pages.roadmap, results: Pages.results, graph: Pages.graph };
if (boot[PAGE]) boot[PAGE]();
window.SkillGraph = { load, scoreAll, explain, extractSkills };
})();
