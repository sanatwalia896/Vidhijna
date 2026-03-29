/**
 * Vidhijna v2 — Frontend Controller
 * Handles: mode switching, SSE streaming, research terminal,
 *          file upload, drafting, side panel, thread memory
 */

"use strict";

// const API = "https://vidhijna-api-122979848414.us-central1.run.app";
const API = "http://localhost:8000";

// ── State ─────────────────────────────────────────────────────────────────────
const S = {
    mode: "auto",
    threadId: uid(),
    streaming: false,
    draftType: "nda",
    file: null,
    fileName: null,
    loops: 3,
    threads: [],
    dragDepth: 0,
    citations: [],
    entities: {},
    loopCount: 0,
};

function uid() { return `t_${crypto.randomUUID().slice(0, 10)}`; }
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];

// ── Mode config ───────────────────────────────────────────────────────────────
const MODES = {
    auto: { glyph: "✦", label: "Auto · AI Supervisor", hint: "Ask about Indian business law…" },
    research: { glyph: "◎", label: "Research · Deep Analysis", hint: "Research statutes, cases, regulations…" },
    chat: { glyph: "◇", label: "Chat · Quick Q&A", hint: "Ask a quick legal question…" },
    document: { glyph: "▭", label: "Document AI · Analysis", hint: "Ask about the uploaded document…" },
    draft: { glyph: "✎", label: "Draft Maker · Generator", hint: "Describe what you need drafted…" },
};

const NODE_META = {
    supervisor: { cls: "p-supervisor", prefix: "SUPERVISOR", label: "Routing intent" },
    generate_query: { cls: "p-research", prefix: "QUERY", label: "Rewriting query" },
    retrieve_legal: { cls: "p-retrieve", prefix: "LEGAL", label: "Fetching legal provisions" },
    retrieve_books: { cls: "p-retrieve", prefix: "BOOKS", label: "Fetching commentary" },
    web_search: { cls: "p-web", prefix: "WEB", label: "Searching live sources" },
    summarize_vectors: { cls: "p-retrieve", prefix: "SUMMARIZE", label: "Summarising provisions" },
    summarize_web: { cls: "p-web", prefix: "SUMMARIZE", label: "Summarising web results" },
    combine: { cls: "p-synthesis", prefix: "COMBINE", label: "Merging research streams" },
    extract_entities: { cls: "p-entity", prefix: "ENTITIES", label: "Extracting legal entities" },
    reflect: { cls: "p-reflect", prefix: "REFLECT", label: "Checking for gaps" },
    finalize: { cls: "p-synthesis", prefix: "FINALIZE", label: "Generating final report" },
    response_formatter: { cls: "p-synthesis", prefix: "FORMAT", label: "Formatting response" },
    validate: { cls: "p-supervisor", prefix: "VALIDATE", label: "Validating inputs" },
    draft: { cls: "p-synthesis", prefix: "DRAFT", label: "Drafting document" },
    review: { cls: "p-reflect", prefix: "REVIEW", label: "Reviewing draft" },
    analyse: { cls: "p-retrieve", prefix: "ANALYSE", label: "Analysing document" },
    flag_risks: { cls: "p-entity", prefix: "RISKS", label: "Flagging risk clauses" },
    answer: { cls: "p-synthesis", prefix: "ANSWER", label: "Generating answer" },
    chat_agent: { cls: "p-supervisor", prefix: "CHAT", label: "Starting chat agent" },
    research_agent: { cls: "p-research", prefix: "RESEARCH", label: "Starting research agent" },
    document_agent: { cls: "p-retrieve", prefix: "DOCUMENT", label: "Starting document agent" },
    draft_agent: { cls: "p-synthesis", prefix: "DRAFT", label: "Starting draft agent" },
};

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    bindModeNav();
    bindDepthControls();
    bindFileUpload();
    bindDragDrop();
    bindNewThread();
    bindSidePanel();
    bindDraftView();
    bindDocView();
    bindMainInput();
    setMode("auto");
});

// ── Mode switching ────────────────────────────────────────────────────────────
function setMode(m) {
    S.mode = m;
    const cfg = MODES[m];
    document.body.setAttribute("data-mode", m);

    // Sidebar highlight
    $$(".mode-item").forEach(b => b.classList.toggle("active", b.dataset.mode === m));

    // Topbar badge
    $("#mb-glyph").textContent = cfg.glyph;
    $("#mb-label").textContent = cfg.label;

    // Show/hide sidebar panels
    toggle("#sb-research", m === "research");
    toggle("#sb-draft", m === "draft");
    toggle("#loop-badge", m === "research");

    // Global input visible for text modes
    toggle("#input-dock", m === "auto" || m === "chat" || m === "research");

    // Update placeholder
    const inp = $("#user-input");
    if (inp) inp.placeholder = cfg.hint;

    // View routing
    if (m === "document") {
        showView("document");
    } else if (m === "draft") {
        showView("draft");
    } else if (hasChatContent()) {
        showView("chat");
    } else {
        showView("welcome");
        renderWelcome(m);
    }
}

function hasChatContent() {
    return ($("#chat-messages")?.children.length ?? 0) > 0;
}

function showView(name) {
    $$(".view").forEach(v => {
        v.classList.remove("active");
        v.classList.add("hidden");
    });
    const el = $(`#view-${name}`);
    if (el) { el.classList.remove("hidden"); el.classList.add("active"); }
}

function toggle(sel, show) {
    const el = $(sel);
    if (el) el.classList.toggle("hidden", !show);
}

// ── Welcome screens ───────────────────────────────────────────────────────────
function renderWelcome(m) {
    const inner = $("#welcome-inner");
    if (!inner) return;
    const builders = { auto: buildWelcomeAuto, research: buildWelcomeResearch, chat: buildWelcomeChat };
    inner.innerHTML = "";
    inner.appendChild((builders[m] || buildWelcomeAuto)());
    // Bind interactions
    inner.querySelectorAll("[data-q]").forEach(el => {
        el.addEventListener("click", () => {
            const mswitch = el.dataset.mswitch;
            if (mswitch) { setMode(mswitch); return; }
            sendMessage(el.dataset.q);
        });
    });
}

function buildWelcomeAuto() {
    return makeWelcome({
        eyebrow: "Multi-Agent · Indian Business Law",
        headline: "Your AI-Powered <em>Legal Research Partner</em>",
        sub: "Research statutes, analyse contracts, draft legal documents — backed by specialist AI agents trained on Indian law.",
        stats: [
            { n: "4", l: "Specialist Agents" },
            { n: "43+", l: "Legal Acts Indexed" },
            { n: "OCR", l: "Document AI" },
            { n: "IBC·SEBI", l: "Commercial Law" },
        ],
        cards: [
            { g: "◎", title: "Deep Research", desc: "Multi-source analysis with reflection loops", q: "What are directors' fiduciary duties under Companies Act 2013?" },
            { g: "▭", title: "Document AI", desc: "Upload contracts, get risk analysis", q: "", ms: "document" },
            { g: "✎", title: "Draft Maker", desc: "Generate NDAs, notices, petitions", q: "", ms: "draft" },
        ],
        chips: [
            { l: "Directors' Duties", q: "What are the directors' duties under Companies Act 2013?" },
            { l: "IBC CIRP Process", q: "Explain the CIRP process under IBC 2016 step by step" },
            { l: "SEBI Insider Trading", q: "What are the penalties for insider trading under SEBI regulations?" },
            { l: "GST Compliance", q: "What are GST compliance requirements for Indian startups?" },
            { l: "Arbitration in India", q: "How does arbitration work under the Arbitration and Conciliation Act 1996?" },
        ],
    });
}

function buildWelcomeResearch() {
    return makeWelcome({
        eyebrow: "Deep Research Mode · Multi-Source Analysis",
        headline: "Scholarly Legal <em>Research Engine</em>",
        sub: "Dual-namespace vector retrieval combined with live regulatory sources. Control reflection depth — more loops means deeper gap-filling analysis.",
        stats: [
            { n: "Pinecone", l: "Vector Store" },
            { n: "Tavily", l: "Live Sources" },
            { n: "Groq", l: "LLM Engine" },
            { n: "3×", l: "Default Loops" },
        ],
        cards: [
            { g: "§", title: "Statutory Analysis", desc: "Section-by-section breakdown of any Indian Act", q: "Explain Section 7 of the Insolvency and Bankruptcy Code 2016" },
            { g: "⚖", title: "Case Law Research", desc: "Supreme Court and High Court judgments", q: "Key Supreme Court judgments on director liability under Companies Act" },
            { g: "◉", title: "Regulatory Updates", desc: "SEBI, RBI, MCA — live circulars and notifications", q: "Latest SEBI regulations on insider trading 2024" },
        ],
        chips: [
            { l: "IBC Section 7", q: "Explain Section 7 financial creditor petition under IBC 2016" },
            { l: "Companies Act 2013", q: "Key provisions of Companies Act 2013 for private limited companies" },
            { l: "SEBI ICDR", q: "What are the SEBI ICDR regulations for IPO disclosure?" },
            { l: "SARFAESI Act", q: "How does the SARFAESI Act 2002 enable banks to recover loans?" },
            { l: "NCLT Jurisdiction", q: "What is the jurisdiction of NCLT under IBC and Companies Act?" },
        ],
    });
}

function buildWelcomeChat() {
    return makeWelcome({
        eyebrow: "Legal Chat · Quick Q&A · Memory-Enabled",
        headline: "Ask Anything About <em>Indian Business Law</em>",
        sub: "Get quick, conversational answers with full context memory. Follow-up questions work naturally — the AI remembers what you discussed.",
        stats: [
            { n: "Memory", l: "Thread Persistence" },
            { n: "Fast", l: "Sub-2s Response" },
            { n: "8B", l: "Groq LLaMA" },
            { n: "All Acts", l: "Commercial Law" },
        ],
        cards: [
            { g: "◇", title: "Corporate Law", desc: "Companies, directors, shareholders, winding up", q: "What is the difference between a private and public limited company in India?" },
            { g: "◈", title: "Contract Law", desc: "Enforceability, breach, remedies, damages", q: "Is a verbal contract legally binding under Indian law?" },
            { g: "◉", title: "Quick Facts", desc: "Compliance deadlines, penalties, procedural steps", q: "What is the penalty for late GST filing in India?" },
        ],
        chips: [
            { l: "Verbal Contract", q: "Is a verbal contract legally binding in India?" },
            { l: "Director Liability", q: "What is the personal liability of a director in a private company?" },
            { l: "Startup Compliance", q: "What are the compliance requirements for a new Indian startup?" },
            { l: "IP Protection", q: "How to protect intellectual property in India?" },
            { l: "FIR vs Complaint", q: "What is the difference between an FIR and a complaint under Indian law?" },
        ],
    });
}

function makeWelcome({ eyebrow, headline, sub, stats, cards, chips }) {
    const frag = document.createElement("div");
    frag.style.cssText = "display:flex;flex-direction:column;align-items:center;width:100%;gap:0";

    const eyeEl = `<div class="wc-eyebrow">${eyebrow}</div>`;
    const headEl = `<h1 class="wc-headline">${headline}</h1>`;
    const subEl = `<p class="wc-sub">${sub}</p>`;

    const statsEl = stats ? `
        <div class="wc-stats">
            ${stats.map(s => `<div class="wc-stat"><span class="wc-stat-n">${s.n}</span><span class="wc-stat-l">${s.l}</span></div>`).join("")}
        </div>` : "";

    const cardsEl = `
        <div class="wc-cards">
            ${cards.map(c => `
                <button class="wc-card" data-q="${esc(c.q)}" ${c.ms ? `data-mswitch="${c.ms}"` : ""}>
                    <div class="wc-card-g">${c.g}</div>
                    <div class="wc-card-title">${c.title}</div>
                    <div class="wc-card-desc">${c.desc}</div>
                </button>`).join("")}
        </div>`;

    const chipsEl = chips ? `
        <div class="wc-chips-wrap">
            <div class="wc-chips-label">Quick prompts</div>
            <div class="wc-chips">
                ${chips.map(ch => `<button class="wc-chip" data-q="${esc(ch.q)}">${ch.l}</button>`).join("")}
            </div>
        </div>` : "";

    frag.innerHTML = eyeEl + headEl + subEl + statsEl + cardsEl + chipsEl;
    return frag;
}

// ── Mode nav ──────────────────────────────────────────────────────────────────
function bindModeNav() {
    $$(".mode-item").forEach(btn =>
        btn.addEventListener("click", () => setMode(btn.dataset.mode))
    );
}

// ── Depth controls ────────────────────────────────────────────────────────────
function bindDepthControls() {
    const slider = $("#depth-slider");
    const valEl = $("#depth-val");
    const badge = $("#loop-badge-label");

    function setDepth(v) {
        S.loops = parseInt(v);
        if (valEl) valEl.textContent = v;
        if (badge) badge.textContent = `${v} loop${v > 1 ? "s" : ""}`;
        if (slider) slider.value = v;
        $$(".dp-btn").forEach(b => b.classList.toggle("active", parseInt(b.dataset.v) === S.loops));
    }

    slider?.addEventListener("input", e => setDepth(e.target.value));
    $$(".dp-btn").forEach(b => b.addEventListener("click", () => setDepth(b.dataset.v)));
    setDepth(3);
}

// ── Main input ────────────────────────────────────────────────────────────────
function bindMainInput() {
    const inp = $("#user-input");
    const btn = $("#send-btn");
    if (!inp || !btn) return;

    btn.addEventListener("click", () => sendMessage());
    inp.addEventListener("keydown", e => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
    inp.addEventListener("input", () => {
        inp.style.height = "auto";
        inp.style.height = Math.min(inp.scrollHeight, 200) + "px";
    });
}

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage(override = null) {
    const inp = $("#user-input");
    const q = override || inp?.value.trim() || "";
    if (!q || S.streaming) return;

    if (inp && !override) { inp.value = ""; inp.style.height = "auto"; }

    showView("chat");
    appendMsg("user", q);
    const aiEl = appendMsg("ai", "", true);

    S.streaming = true;
    setDisabled(true);
    S.citations = [];
    S.entities = {};
    S.loopCount = 0;

    const isResearch = S.mode === "research" || S.mode === "auto";

    if (isResearch) {
        showRT();
        rtLog("p-supervisor", "INIT", "Starting Vidhijna agent pipeline…");
        updateRTPhase("Initializing");
        openSidePanel();
        clearSidePanel();
    }

    try {
        const res = await fetch(`${API}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: q,
                thread_id: S.threadId,
                mode: S.mode,
                draft_type: S.draftType,
                draft_inputs: {},
                reflection_loops: S.loops,
            }),
        });

        if (!res.ok) throw new Error(`Server error ${res.status}`);

        await consumeSSE(res, aiEl, isResearch);

    } catch (err) {
        setMsgContent(aiEl, `⚠️ **Connection error:** ${err.message}\n\nMake sure the backend is running at \`${API}\`.`);
        if (isResearch) rtLog("p-error", "ERROR", err.message);
    } finally {
        removeCursor(aiEl);
        S.streaming = false;
        setDisabled(false);
        if (isResearch) { rtLog("p-synthesis", "DONE", "Pipeline complete."); updateRTPhase("Done"); }
        saveThread(q);
    }
}

// ── SSE consumer ──────────────────────────────────────────────────────────────
async function consumeSSE(res, msgEl, isResearch) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";

        for (const part of parts) {
            if (!part.startsWith("data:")) continue;
            try {
                const evt = JSON.parse(part.slice(5).trim());
                handleEvt(evt, msgEl, isResearch);
            } catch { /* ignore malformed */ }
        }
    }
}

function handleEvt(evt, msgEl, isResearch) {
    switch (evt.type) {

        case "status":
            if (isResearch && evt.content) rtLog("p-supervisor", "INFO", evt.content);
            if (!msgEl.dataset.hasContent) setLoadingMsg(msgEl, evt.content || "Working…");
            break;

        case "node_start": {
            const m = NODE_META[evt.node] || {};
            if (isResearch) {
                rtLog(m.cls || "p-supervisor", m.prefix || evt.node.toUpperCase(), m.label || evt.content || "");
                updateRTPhase(m.label || evt.content || "");
            }
            break;
        }

        case "loop_start":
            if (isResearch) {
                S.loopCount++;
                rtLoopDivider(S.loopCount, S.loops);
                updateRTPhase(`Reflection loop ${S.loopCount} / ${S.loops}`);
            }
            break;

        case "research_card":
            if (isResearch) {
                rtCard(evt.icon || "◎", evt.category || "", evt.content || "");
                addFinding(evt.icon || "◎", evt.category || "", evt.content || "");
            }
            break;

        case "citations":
            if (evt.items) {
                S.citations.push(...evt.items);
                evt.items.forEach(c => addSource(c));
            }
            break;

        case "legal_entities":
            if (evt.entities) {
                Object.assign(S.entities, evt.entities);
                updateEntities(evt.entities);
            }
            break;

        case "risk_flag":
            addRiskChip(evt.content || "", evt.severity || "medium");
            break;

        case "token":
            appendToken(msgEl, evt.content || "");
            msgEl.dataset.hasContent = "1";
            break;

        case "final":
            if (evt.content) {
                setMsgContent(msgEl, evt.content);
                msgEl.dataset.hasContent = "1";
                if (evt.citations) { S.citations.push(...evt.citations); evt.citations.forEach(c => addSource(c)); }
                if (evt.entities) { Object.assign(S.entities, evt.entities); updateEntities(evt.entities); }
                if (isResearch) rtLog("p-synthesis", "FINAL", "Response ready.");
            }
            break;

        case "error":
            setMsgContent(msgEl, `⚠️ ${evt.content || "An error occurred."}`);
            if (isResearch) rtLog("p-error", "ERROR", evt.content || "");
            break;
    }
}

// ── Message helpers ───────────────────────────────────────────────────────────
function appendMsg(role, content, streaming = false) {
    const ctr = $("#chat-messages");
    if (!ctr) return null;
    const m = MODES[S.mode];
    const el = document.createElement("div");
    el.className = `msg ${role}`;

    el.innerHTML = `
        <div class="msg-ava">${role === "user" ? "You" : "VJ"}</div>
        <div class="msg-body">
            <div class="msg-bubble">${streaming && !content
            ? `<span class="loading-dots"></span>`
            : md(content)
        }${streaming ? `<span class="typing-cursor"></span>` : ""}</div>
            <div class="msg-meta">
                ${role === "ai" ? `<span class="msg-badge">${m?.glyph} ${m?.label}</span>` : ""}
                <span>${fmtTime(new Date())}</span>
            </div>
        </div>`;

    ctr.appendChild(el);
    ctr.scrollTop = ctr.scrollHeight;
    return el;
}

function setMsgContent(el, text) {
    const b = el?.querySelector(".msg-bubble");
    if (b) b.innerHTML = md(text);
}

function setLoadingMsg(el, text) {
    if (!el || el.dataset.hasContent) return;
    const b = el.querySelector(".msg-bubble");
    if (b) b.innerHTML = `<span style="color:var(--ink-400);font-style:italic;font-size:0.88rem">${esc(text)}</span><span class="typing-cursor"></span>`;
}

function appendToken(el, tok) {
    const b = el?.querySelector(".msg-bubble");
    if (!b) return;
    const cursor = b.querySelector(".typing-cursor");
    const s = document.createElement("span");
    s.textContent = tok;
    if (cursor) b.insertBefore(s, cursor);
    else b.appendChild(s);
    el.closest(".chat-messages")?.scrollTo({ top: 99999 });
}

function removeCursor(el) {
    el?.querySelector(".typing-cursor")?.remove();
    el?.classList.remove("msg-streaming");
}

// ── Research terminal ─────────────────────────────────────────────────────────
function showRT() {
    const rt = $("#rt");
    if (rt) rt.classList.remove("hidden");
    clearRT();
}

function rtLog(cls, prefix, msg) {
    const body = $("#rt-body");
    if (!body) return;
    const now = new Date();
    const ts = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    const line = document.createElement("div");
    line.className = "rt-line";
    line.innerHTML = `<span class="rt-ts">[${ts}]</span><span class="rt-prefix ${cls}">${prefix}</span><span class="rt-msg">${escH(msg)}</span>`;
    body.appendChild(line);
    body.scrollTop = body.scrollHeight;
}

function rtLoopDivider(n, total) {
    const body = $("#rt-body");
    if (!body) return;
    const d = document.createElement("div");
    d.className = "rt-loop-line";
    d.textContent = `Loop ${n} / ${total}`;
    body.appendChild(d);
    body.scrollTop = body.scrollHeight;
}

function rtCard(icon, cat, content) {
    const body = $("#rt-body");
    if (!body) return;
    const c = document.createElement("div");
    c.className = "rt-card";
    c.innerHTML = `<div class="rt-card-hdr">${icon} ${escH(cat)}</div><div class="rt-card-body">${escH(content.slice(0, 220))}${content.length > 220 ? "…" : ""}</div>`;
    body.appendChild(c);
    body.scrollTop = body.scrollHeight;
}

function clearRT() {
    const body = $("#rt-body");
    if (body) body.innerHTML = "";
}

function updateRTPhase(label) {
    const el = $("#rt-phase");
    if (el) el.textContent = label;
}

// RT minimize
document.addEventListener("DOMContentLoaded", () => {
    let collapsed = false;
    $("#rt-min")?.addEventListener("click", () => {
        collapsed = !collapsed;
        const body = $("#rt-body");
        if (body) body.style.display = collapsed ? "none" : "";
        const btn = $("#rt-min");
        if (btn) btn.textContent = collapsed ? "+" : "−";
    });
});

// ── Side panel ────────────────────────────────────────────────────────────────
function bindSidePanel() {
    $$(".sp-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            $$(".sp-tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            const name = tab.dataset.tab;
            $$(".sp-panel").forEach(p => p.classList.toggle("hidden", p.id !== `sp-${name}`));
        });
    });
    $("#sp-close")?.addEventListener("click", closeSidePanel);
}

function openSidePanel() {
    $("#side-panel")?.classList.remove("hidden");
}
function closeSidePanel() {
    $("#side-panel")?.classList.add("hidden");
}
function clearSidePanel() {
    const f = $("#sp-findings");
    const e = $("#sp-entities");
    const s = $("#sp-sources");
    if (f) f.innerHTML = `<div class="sp-empty"><span class="sp-empty-glyph">◎</span><p>Research findings will appear here as the agent works</p></div>`;
    if (e) e.innerHTML = `<div class="sp-empty"><span class="sp-empty-glyph">▦</span><p>Extracted statutes, cases, and legal entities</p></div>`;
    if (s) s.innerHTML = `<div class="sp-empty"><span class="sp-empty-glyph">▤</span><p>Citations and source documents</p></div>`;
}

function addFinding(icon, cat, content) {
    const p = $("#sp-findings");
    if (!p) return;
    p.querySelector(".sp-empty")?.remove();
    const c = document.createElement("div");
    c.className = "sp-card";
    c.innerHTML = `<div class="sp-card-hdr"><span>${icon}</span>${escH(cat)}</div><div class="sp-card-body">${escH(content.slice(0, 300))}${content.length > 300 ? "…" : ""}</div>`;
    p.appendChild(c);
}

function addSource(cit) {
    const p = $("#sp-sources");
    if (!p) return;
    p.querySelector(".sp-empty")?.remove();
    const d = document.createElement("div");
    d.className = "source-item";
    d.textContent = cit;
    p.appendChild(d);
}

function updateEntities(ents) {
    const p = $("#sp-entities");
    if (!p) return;
    p.querySelector(".sp-empty")?.remove();
    Object.entries(ents).forEach(([cat, items]) => {
        if (!items?.length) return;
        const g = document.createElement("div");
        g.className = "entity-group";
        g.innerHTML = `<div class="entity-group-label">${escH(cat)}</div>
            <div class="entity-tags">${items.slice(0, 8).map(i => `<span class="entity-tag">${escH(String(i))}</span>`).join("")}</div>`;
        p.appendChild(g);
    });
}

// ── Risk flags ────────────────────────────────────────────────────────────────
function addRiskChip(content, severity) {
    const bar = $("#doc-risk-bar");
    if (!bar) return;
    const chip = document.createElement("span");
    chip.className = `risk-chip ${severity}`;
    chip.textContent = content.slice(0, 55);
    bar.appendChild(chip);
}

// ── Document view ─────────────────────────────────────────────────────────────
function bindDocView() {
    const drop = $("#doc-drop");
    drop?.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("dragover"); });
    drop?.addEventListener("dragleave", () => drop.classList.remove("dragover"));
    drop?.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("dragover"); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });

    // Doc info tabs
    $$(".di-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            $$(".di-tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            const name = tab.dataset.dtab;
            $$(".di-panel").forEach(p => p.classList.toggle("hidden", p.id !== `dt-${name}`));
        });
    });

    // Doc input send
    const docInp = $("#doc-query");
    const docBtn = $("#doc-send");
    docBtn?.addEventListener("click", () => sendDocMsg());
    docInp?.addEventListener("keydown", e => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendDocMsg(); }
    });
    docInp?.addEventListener("input", () => {
        docInp.style.height = "auto";
        docInp.style.height = Math.min(docInp.scrollHeight, 120) + "px";
    });
}

function handleFile(file) {
    S.file = file;
    S.fileName = file.name;

    // Switch to document mode + analysis view
    if (S.mode !== "document") setMode("document");

    const upload = $("#doc-upload");
    const analysis = $("#doc-analysis");
    if (upload) upload.classList.add("hidden");
    if (analysis) { analysis.classList.remove("hidden"); analysis.classList.add("flex"); }

    // File info bar
    const bar = $("#doc-file-bar");
    if (bar) {
        bar.innerHTML = `
            <div class="dfb-icon">▭</div>
            <div>
                <div class="dfb-name">${escH(file.name.length > 40 ? file.name.slice(0, 38) + "…" : file.name)}</div>
                <div class="dfb-meta">${fmtSize(file.size)} · Ready for analysis</div>
            </div>`;
    }

    // Auto-analyse
    setTimeout(() => sendDocMsg("Analyse this document for risks, key clauses, obligations, and compliance issues under Indian law."), 300);
}

async function sendDocMsg(override = null) {
    const inp = $("#doc-query");
    const q = override || inp?.value.trim() || "";
    if (!q || !S.file || S.streaming) return;

    if (inp && !override) { inp.value = ""; inp.style.height = "auto"; }

    const ctr = $("#doc-messages");
    if (!ctr) return;

    appendMsgTo(ctr, "user", q);
    const aiEl = appendMsgTo(ctr, "ai", "", true);
    S.streaming = true;
    const docBtn = $("#doc-send");
    if (docBtn) docBtn.disabled = true;

    try {
        const form = new FormData();
        form.append("file", S.file);
        form.append("thread_id", S.threadId);
        form.append("query", q);

        const res = await fetch(`${API}/upload`, { method: "POST", body: form });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        await consumeSSE(res, aiEl, false);

    } catch (err) {
        setMsgContent(aiEl, `⚠️ ${err.message}`);
    } finally {
        removeCursor(aiEl);
        S.streaming = false;
        if (docBtn) docBtn.disabled = false;
    }
}

function appendMsgTo(ctr, role, content, streaming = false) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.innerHTML = `
        <div class="msg-ava">${role === "user" ? "You" : "VJ"}</div>
        <div class="msg-body">
            <div class="msg-bubble">${streaming && !content
            ? `<span class="loading-dots"></span>`
            : md(content)
        }${streaming ? `<span class="typing-cursor"></span>` : ""}</div>
        </div>`;
    ctr.appendChild(el);
    ctr.scrollTop = ctr.scrollHeight;
    return el;
}

// ── Draft view ────────────────────────────────────────────────────────────────
function bindDraftView() {
    $$(".df-tpl").forEach(btn => {
        btn.addEventListener("click", () => {
            $$(".df-tpl").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            S.draftType = btn.dataset.draft;
            const sel = $("#draft-type-select");
            if (sel) sel.value = S.draftType;
        });
    });

    $("#draft-type-select")?.addEventListener("change", e => {
        S.draftType = e.target.value;
        $$(".df-tpl").forEach(b => b.classList.toggle("active", b.dataset.draft === S.draftType));
    });

    $("#btn-generate")?.addEventListener("click", generateDraft);

    // Copy
    $("#dp-copy")?.addEventListener("click", () => {
        const body = $("#dp-content");
        if (body) navigator.clipboard.writeText(body.innerText);
        const btn = $("#dp-copy");
        const orig = btn.textContent;
        btn.textContent = "✓ Copied";
        setTimeout(() => { btn.textContent = orig; }, 2000);
    });

    // Download
    $("#dp-download")?.addEventListener("click", () => {
        const body = $("#dp-content");
        if (!body) return;
        const blob = new Blob([body.innerText], { type: "text/plain" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `${S.draftType}_vidhijna.txt`;
        a.click();
    });
}

async function generateDraft() {
    const partyA = $("#df-party-a")?.value.trim() || "";
    const partyB = $("#df-party-b")?.value.trim() || "";
    const instr = $("#df-instructions")?.value.trim() || "";

    const q = `Draft a ${S.draftType.replace(/_/g, " ")}${partyA ? ` between ${partyA}` : ""}${partyB ? ` and ${partyB}` : ""}${instr ? `. Requirements: ${instr}` : ""}. Provide a complete, legally sound document with all standard clauses for Indian jurisdiction.`;

    const btn = $("#btn-generate");
    const content = $("#dp-content");
    const title = $("#dp-title");
    const status = $("#dp-status");

    if (btn) btn.disabled = true;
    if (title) title.textContent = draftLabel(S.draftType);
    if (status) status.textContent = "Generating…";

    // Skeleton
    if (content) {
        content.innerHTML = `<div class="draft-skel">${[90, 72, 85, 65, 92, 78, 88, 60].map((w, i) =>
            `<div class="skel-line" style="width:${w}%;animation-delay:${i * 0.08}s"></div>`
        ).join("")
            }</div>`;
    }

    S.streaming = true;

    try {
        const res = await fetch(`${API}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: q,
                thread_id: S.threadId,
                mode: "draft",
                draft_type: S.draftType,
                draft_inputs: { party_a: partyA, party_b: partyB, instructions: instr },
            }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            const parts = buf.split("\n\n");
            buf = parts.pop() ?? "";
            for (const part of parts) {
                if (!part.startsWith("data:")) continue;
                try {
                    const evt = JSON.parse(part.slice(5).trim());
                    if (evt.type === "status" && status) status.textContent = evt.content;
                    if (evt.type === "final" && evt.content && content) {
                        content.innerHTML = `<div class="legal-doc">${md(evt.content)}</div>`;
                        if (status) status.textContent = `Generated ${fmtTime(new Date())} · ${draftLabel(S.draftType)}`;
                    }
                } catch { }
            }
        }
    } catch (err) {
        if (content) content.innerHTML = `<div class="dp-empty"><div class="dp-empty-glyph">⚠</div><p>${err.message}</p></div>`;
        if (status) status.textContent = "Generation failed";
    } finally {
        S.streaming = false;
        if (btn) btn.disabled = false;
    }
}

function draftLabel(type) {
    const MAP = {
        nda: "Non-Disclosure Agreement", service_agreement: "Service Agreement",
        employment: "Employment Contract", lease: "Lease Agreement",
        legal_notice: "Legal Notice", cease_desist: "Cease & Desist",
        nclt_petition: "NCLT Petition", arbitration_notice: "Arbitration Notice",
        consumer_complaint: "Consumer Complaint", reply_notice: "Reply to Notice",
    };
    return MAP[type] || type.replace(/_/g, " ");
}

// ── File upload ───────────────────────────────────────────────────────────────
function bindFileUpload() {
    const inp = $("#file-input");
    inp?.addEventListener("change", e => { if (e.target.files[0]) handleFile(e.target.files[0]); });
}

// ── Drag and drop ─────────────────────────────────────────────────────────────
function bindDragDrop() {
    const overlay = $("#drop-overlay");

    document.addEventListener("dragenter", e => {
        e.preventDefault(); S.dragDepth++;
        overlay?.classList.add("active");
    });
    document.addEventListener("dragleave", () => {
        S.dragDepth = Math.max(0, S.dragDepth - 1);
        if (!S.dragDepth) overlay?.classList.remove("active");
    });
    document.addEventListener("dragover", e => e.preventDefault());
    document.addEventListener("drop", e => {
        e.preventDefault(); S.dragDepth = 0;
        overlay?.classList.remove("active");
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });
}

// ── New thread ────────────────────────────────────────────────────────────────
function bindNewThread() {
    $("#btn-new")?.addEventListener("click", newThread);
}

function newThread() {
    S.threadId = uid();
    S.streaming = false;
    S.citations = [];
    S.entities = {};
    S.file = null;
    S.loopCount = 0;

    const chatMsgs = $("#chat-messages");
    const docMsgs = $("#doc-messages");
    const rtBody = $("#rt-body");
    if (chatMsgs) chatMsgs.innerHTML = "";
    if (docMsgs) docMsgs.innerHTML = "";
    if (rtBody) rtBody.innerHTML = "";
    $("#rt")?.classList.add("hidden");

    // Reset doc view
    $("#doc-upload")?.classList.remove("hidden");
    $("#doc-analysis")?.classList.add("hidden");

    // Reset draft
    const dpc = $("#dp-content");
    if (dpc) dpc.innerHTML = `<div class="dp-empty"><div class="dp-empty-glyph">✎</div><p>Your generated document will appear here</p></div>`;

    closeSidePanel();
    clearSidePanel();
    setDisabled(false);
    showView("welcome");
    renderWelcome(S.mode);

    const tbThread = $("#tb-thread");
    if (tbThread) tbThread.textContent = `#${S.threadId.slice(2, 10)}`;

    renderThreadList();
}

// ── Thread list ───────────────────────────────────────────────────────────────
function saveThread(query) {
    const existing = S.threads.find(t => t.id === S.threadId);
    if (existing) {
        existing.last = query; existing.time = new Date();
    } else {
        S.threads.unshift({
            id: S.threadId,
            title: query.slice(0, 48) + (query.length > 48 ? "…" : ""),
            mode: S.mode, time: new Date(),
        });
        if (S.threads.length > 25) S.threads.pop();
    }
    const tb = $("#tb-thread");
    if (tb) tb.textContent = `#${S.threadId.slice(2, 10)}`;
    renderThreadList();
}

function renderThreadList() {
    const list = $("#thread-list");
    if (!list) return;
    if (!S.threads.length) {
        list.innerHTML = `<div class="thread-empty">No conversations yet</div>`;
        return;
    }
    list.innerHTML = S.threads.map(t => `
        <div class="thread-item${t.id === S.threadId ? " active" : ""}" data-tid="${t.id}">
            <div class="ti-title">${escH(t.title)}</div>
            <div class="ti-meta">${MODES[t.mode]?.glyph ?? "✦"} ${fmtTime(t.time)}</div>
        </div>`).join("");

    list.querySelectorAll(".thread-item").forEach(item => {
        item.addEventListener("click", () => {
            S.threadId = item.dataset.tid;
            renderThreadList();
            const tb = $("#tb-thread");
            if (tb) tb.textContent = `#${S.threadId.slice(2, 10)}`;
        });
    });
}

// ── Health check (runs once on demand, no polling) ───────────────────────────
async function healthCheck() {
    const dot = $("#sb-status-dot");
    const label = $("#sb-status-text");
    try {
        const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(5000) });
        const data = await res.json();
        dot?.classList.add("online"); dot?.classList.remove("offline");
        if (label) label.textContent = data.status === "healthy" ? "System online" : "Degraded";
    } catch {
        dot?.classList.add("offline"); dot?.classList.remove("online");
        if (label) label.textContent = "Backend offline";
    }
}

// ── Misc helpers ──────────────────────────────────────────────────────────────
function setDisabled(v) {
    const btns = [$("#send-btn"), $("#doc-send"), $("#btn-generate")];
    btns.forEach(b => { if (b) b.disabled = v; });
}

function md(text) {
    if (!text) return "";
    try { return typeof marked !== "undefined" ? marked.parse(text) : escH(text).replace(/\n/g, "<br>"); }
    catch { return escH(text).replace(/\n/g, "<br>"); }
}

function esc(str) { return (str || "").replace(/"/g, "&quot;").replace(/'/g, "&#39;"); }
function escH(str) { return (str || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
function fmtTime(d) { return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
function fmtSize(b) { return b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`; }
function pad(n) { return String(n).padStart(2, "0"); }