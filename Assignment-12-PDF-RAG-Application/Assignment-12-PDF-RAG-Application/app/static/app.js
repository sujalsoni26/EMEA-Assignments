"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  fileInput: $("file-input"), dropzone: $("dropzone"), uploadStatus: $("upload-status"), limitsHint: $("limits-hint"),
  docList: $("doc-list"), docEmpty: $("doc-empty"), kbStats: $("kb-stats"), clearBtn: $("clear-btn"),
  messages: $("messages"), welcome: $("welcome"), examples: $("examples"),
  form: $("ask-form"), question: $("question"), askBtn: $("ask-btn"), topK: $("top-k"), topKValue: $("top-k-value"),
  status: $("status"), statusText: $("status-text"), template: $("answer-template"),
};

const config = { maxFileMb: 25, maxFiles: 10, maxQuestionChars: 1000 };
const EXAMPLES = [
  "How many days of annual leave do full-time employees get?",
  "How often should the RO membrane be replaced?",
  "What are the four functions of the AI RMF Core?",
  "What does error code E2 mean?",
  "What is the capital of Australia?",
];
let documentCount = 0;
let asking = false;

const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const formatSize = (b) => (b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);

async function readError(response) {
  try {
    const data = await response.json();
    return (data.error && data.error.message) || `Request failed (${response.status}).`;
  } catch {
    return `Request failed (${response.status}).`;
  }
}

// ------------------------------------------------------------------ knowledge base
async function loadDocuments() {
  try {
    const response = await fetch("/api/documents");
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json();
    documentCount = data.documents.length;
    els.docList.replaceChildren(...data.documents.map(renderDocument));
    els.docEmpty.hidden = documentCount > 0;
    els.clearBtn.hidden = documentCount === 0;
    els.kbStats.textContent = documentCount
      ? `${documentCount} document${documentCount === 1 ? "" : "s"} · ${data.total_chunks} searchable passages`
      : "";
  } catch (error) {
    els.kbStats.textContent = `Could not load documents: ${error.message}`;
  }
  updateAskButton();
}

function renderDocument(doc) {
  const li = el("li", null, "doc");
  li.appendChild(el("span", "PDF", "doc__icon"));
  const info = el("div", null, "doc__info");
  info.appendChild(el("span", doc.filename, "doc__name"));
  info.appendChild(el("span", `${doc.pages} pages · ${doc.chunks} passages · ${formatSize(doc.size_bytes)}`, "doc__meta"));
  li.appendChild(info);
  const remove = el("button", "✕", "icon-btn");
  remove.type = "button";
  remove.title = `Remove ${doc.filename}`;
  remove.setAttribute("aria-label", `Remove ${doc.filename}`);
  remove.addEventListener("click", () => deleteDocument(doc));
  li.appendChild(remove);
  li.title = doc.filename;
  return li;
}

async function deleteDocument(doc) {
  const response = await fetch(`/api/documents/${encodeURIComponent(doc.id)}`, { method: "DELETE" });
  if (!response.ok && response.status !== 404) {
    showUploadItems([{ filename: doc.filename, status: "error", message: await readError(response) }]);
  }
  loadDocuments();
}

function showUploadItems(items) {
  els.uploadStatus.replaceChildren(...items.map((item) => {
    const div = el("div", null, `upload-item upload-item--${item.status}`);
    const name = el("strong", item.filename);
    div.appendChild(name);
    div.appendChild(document.createTextNode(` — ${item.message}`));
    return div;
  }));
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList);
  if (!files.length) return;
  if (files.length > config.maxFiles) {
    showUploadItems([{ filename: `${files.length} files`, status: "error", message: `Please upload at most ${config.maxFiles} files at a time.` }]);
    return;
  }
  // Client-side checks give instant feedback; the server validates again.
  const valid = [];
  const rejected = [];
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith(".pdf")) rejected.push({ filename: file.name, status: "error", message: "Not a PDF file." });
    else if (file.size === 0) rejected.push({ filename: file.name, status: "error", message: "The file is empty." });
    else if (file.size > config.maxFileMb * 1048576) rejected.push({ filename: file.name, status: "error", message: `Larger than ${config.maxFileMb} MB.` });
    else valid.push(file);
  }
  showUploadItems([...rejected, ...valid.map((f) => ({ filename: f.name, status: "pending", message: "Extracting, chunking and embedding…" }))]);
  if (!valid.length) return;

  const body = new FormData();
  valid.forEach((file) => body.append("files", file));
  els.dropzone.classList.add("busy");
  try {
    const response = await fetch("/api/documents", { method: "POST", body });
    const data = await response.json().catch(() => ({}));
    if (data.results) {
      showUploadItems([...rejected, ...data.results]);
    } else {
      const message = (data.error && data.error.message) || `Upload failed (${response.status}).`;
      showUploadItems([...rejected, ...valid.map((f) => ({ filename: f.name, status: "error", message }))]);
    }
  } catch {
    showUploadItems(valid.map((f) => ({ filename: f.name, status: "error", message: "Cannot reach the server." })));
  } finally {
    els.dropzone.classList.remove("busy");
    els.fileInput.value = "";
    loadDocuments();
  }
}

// ------------------------------------------------------------------ answers
function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function addQuestion(text) {
  els.welcome.hidden = true;
  els.messages.appendChild(el("div", text, "msg msg--question"));
  const pending = el("div", null, "msg msg--answer msg--pending");
  pending.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>Searching the documents and writing an answer…';
  els.messages.appendChild(pending);
  scrollToBottom();
  return pending;
}

/** Render answer text: paragraphs, "- " bullet lists, and [n] citations as buttons. */
function renderAnswerText(container, text, onCite) {
  const blocks = text.split(/\n\s*\n/);
  for (const block of blocks) {
    const lines = block.split("\n").filter((l) => l.trim());
    const isList = lines.length > 0 && lines.every((l) => /^\s*([-*•]|\d+[.)])\s+/.test(l));
    if (isList) {
      const ul = el("ul");
      lines.forEach((line) => {
        const li = el("li");
        appendWithCitations(li, line.replace(/^\s*([-*•]|\d+[.)])\s+/, ""), onCite);
        ul.appendChild(li);
      });
      container.appendChild(ul);
    } else {
      const p = el("p");
      appendWithCitations(p, lines.join(" "), onCite);
      container.appendChild(p);
    }
  }
}

function appendWithCitations(node, text, onCite) {
  const parts = text.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);
  for (const part of parts) {
    const match = part.match(/^\[(\d+(?:\s*,\s*\d+)*)\]$/);
    if (!match) { node.appendChild(document.createTextNode(part)); continue; }
    match[1].split(",").map((n) => parseInt(n, 10)).forEach((n) => {
      const button = el("button", String(n), "cite");
      button.type = "button";
      button.title = `Show source ${n}`;
      button.addEventListener("click", () => onCite(n));
      node.appendChild(button);
    });
  }
}

function renderAnswer(pending, data) {
  const article = els.template.content.firstElementChild.cloneNode(true);
  const body = article.querySelector(".answer__body");
  const sourcesBox = article.querySelector(".sources");
  const list = article.querySelector(".sources__list");

  if (data.status !== "answered") {
    article.classList.add("msg--insufficient");
    body.appendChild(el("span", data.status === "no_documents" ? "No documents" : "Not found in documents", "badge badge--warn"));
  }

  const sourceItems = data.sources.map((source) => {
    const li = el("li", null, `source${source.relevant ? "" : " source--irrelevant"}`);
    const head = el("div", null, "source__head");
    head.appendChild(el("span", `[${source.rank}] ${source.filename} — page ${source.page}`, "source__title"));
    const tags = el("span", null, "source__tags");
    if (source.cited) tags.appendChild(el("span", "cited", "tag tag--cited"));
    if (source.keyword_match && source.keyword_match.length) tags.appendChild(el("span", `keyword: ${source.keyword_match.join(", ")}`, "tag"));
    if (!source.relevant) tags.appendChild(el("span", "below threshold", "tag"));
    const score = el("span", null, "score");
    const bar = el("span", null, "score__bar");
    const fill = el("span");
    fill.style.width = `${Math.max(0, Math.min(1, source.score)) * 100}%`;
    bar.appendChild(fill);
    score.appendChild(bar);
    score.appendChild(document.createTextNode(source.score.toFixed(2)));
    tags.appendChild(score);
    head.appendChild(tags);
    li.appendChild(head);
    li.appendChild(el("p", source.text, "source__text"));
    return li;
  });
  list.replaceChildren(...sourceItems);

  const onCite = (n) => {
    const item = sourceItems[n - 1];
    if (!item) return;
    sourcesBox.open = true;
    item.scrollIntoView({ behavior: "smooth", block: "nearest" });
    item.classList.add("flash");
    setTimeout(() => item.classList.remove("flash"), 1200);
  };
  renderAnswerText(body, data.answer, onCite);

  const used = data.sources.filter((s) => s.relevant).length;
  article.querySelector(".answer__meta").textContent =
    `${data.model} · retrieval ${data.retrieval_ms} ms · generation ${(data.generation_ms / 1000).toFixed(1)} s · ` +
    `${used} of ${data.sources.length} passages sent to the model`;
  if (data.sources.length) {
    sourcesBox.querySelector("summary").textContent = `Retrieved context (${data.sources.length} passages)`;
    if (data.status !== "answered") sourcesBox.querySelector("summary").textContent = `Closest passages found (${data.sources.length})`;
  } else {
    sourcesBox.hidden = true;
  }
  pending.replaceWith(article);
  scrollToBottom();
}

function renderError(pending, message) {
  const div = el("div", null, "msg msg--answer msg--error");
  div.appendChild(el("span", "Error", "badge badge--error"));
  div.appendChild(el("p", message));
  pending.replaceWith(div);
  scrollToBottom();
}

async function ask(question) {
  question = question.trim();
  if (!question || asking) return;
  if (question.length > config.maxQuestionChars) return;
  asking = true;
  updateAskButton();
  els.askBtn.classList.add("loading");
  els.question.value = "";
  autoGrow();
  const pending = addQuestion(question);
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: parseInt(els.topK.value, 10) }),
    });
    if (!response.ok) {
      renderError(pending, await readError(response));
      if (response.status === 503) checkHealth();
      return;
    }
    renderAnswer(pending, await response.json());
  } catch {
    renderError(pending, "Cannot reach the server. Make sure the FastAPI app is running.");
  } finally {
    asking = false;
    els.askBtn.classList.remove("loading");
    updateAskButton();
    els.question.focus();
  }
}

function updateAskButton() {
  els.askBtn.disabled = asking || els.question.value.trim().length < 3;
}

function autoGrow() {
  els.question.style.height = "auto";
  els.question.style.height = `${Math.min(els.question.scrollHeight, 160)}px`;
}

// ------------------------------------------------------------------ health & config
async function checkHealth() {
  els.status.className = "status status--checking";
  els.statusText.textContent = "Checking Ollama…";
  try {
    const data = await (await fetch("/api/health")).json();
    const cls = { ok: "status--ok", degraded: "status--warn" }[data.status] || "status--error";
    els.status.className = `status ${cls}`;
    els.statusText.textContent = data.status === "ok" ? `${data.llm_model} + ${data.embedding_model} ready`
      : data.status === "degraded" ? "Model missing" : "Ollama offline";
    els.status.title = `${data.message} (click to re-check)`;
  } catch {
    els.status.className = "status status--error";
    els.statusText.textContent = "Server offline";
  }
}

async function loadConfig() {
  try {
    const data = await (await fetch("/api/config")).json();
    config.maxFileMb = data.max_file_mb;
    config.maxFiles = data.max_files_per_upload;
    config.maxQuestionChars = data.max_question_chars;
    els.topK.max = data.max_top_k;
    els.topK.value = data.top_k;
    els.topKValue.textContent = data.top_k;
    els.question.maxLength = data.max_question_chars;
    els.limitsHint.textContent = `One or more PDF files · up to ${data.max_file_mb} MB each`;
  } catch { /* keep defaults */ }
}

// ------------------------------------------------------------------ events
els.fileInput.addEventListener("change", () => uploadFiles(els.fileInput.files));
["dragenter", "dragover"].forEach((t) => els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((t) => els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.remove("dragover"); }));
els.dropzone.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));
els.clearBtn.addEventListener("click", async () => {
  await fetch("/api/documents", { method: "DELETE" });
  els.uploadStatus.replaceChildren();
  loadDocuments();
});
els.form.addEventListener("submit", (e) => { e.preventDefault(); ask(els.question.value); });
els.question.addEventListener("input", () => { updateAskButton(); autoGrow(); });
els.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(els.question.value); }
});
els.topK.addEventListener("input", () => { els.topKValue.textContent = els.topK.value; });
els.status.addEventListener("click", checkHealth);
EXAMPLES.forEach((q) => {
  const chip = el("button", q, "chip");
  chip.type = "button";
  chip.addEventListener("click", () => ask(q));
  els.examples.appendChild(chip);
});

loadConfig();
checkHealth();
loadDocuments();
