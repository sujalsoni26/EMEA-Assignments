"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  form: $("form"),
  fileInput: $("file-input"),
  dropzone: $("dropzone"),
  fileCard: $("file-card"),
  fileName: $("file-name"),
  fileSize: $("file-size"),
  fileError: $("file-error"),
  removeFile: $("remove-file"),
  limitsHint: $("limits-hint"),
  submit: $("submit-btn"),
  progressCard: $("progress-card"),
  steps: $("steps"),
  elapsed: $("elapsed"),
  cancel: $("cancel-btn"),
  status: $("status"),
  statusText: $("status-text"),
  empty: $("empty"),
  error: $("error"),
  errorTitle: $("error-title"),
  errorMessage: $("error-message"),
  result: $("result"),
  copy: $("copy-btn"),
  download: $("download-btn"),
};

const limits = { maxFileMb: 20, maxPages: 300 };
let selectedFile = null;
let controller = null;
let timer = null;
let lastResult = null;

const ERROR_TITLES = {
  no_file: "No file selected",
  invalid_file_type: "Not a PDF",
  empty_file: "Empty file",
  file_too_large: "File too large",
  corrupt_pdf: "Could not read the PDF",
  encrypted_pdf: "Password-protected PDF",
  too_many_pages: "Too many pages",
  no_extractable_text: "No readable text",
  validation_error: "Invalid request",
  ollama_unavailable: "Ollama is not reachable",
  model_not_found: "Model not installed",
  model_timeout: "The model took too long",
  invalid_model_output: "The model returned an invalid summary",
  model_error: "Model error",
  internal_error: "Server error",
};

function formatSize(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

// ------------------------------------------------------------ file selection
function validateFile(file) {
  if (!file) return "Please choose a PDF file.";
  const isPdf = file.name.toLowerCase().endsWith(".pdf");
  if (!isPdf) return `"${file.name}" is not a PDF. Please choose a file with the .pdf extension.`;
  if (file.size === 0) return `"${file.name}" is empty.`;
  if (file.size > limits.maxFileMb * 1024 * 1024) {
    return `"${file.name}" is ${formatSize(file.size)}. The maximum size is ${limits.maxFileMb} MB.`;
  }
  return null;
}

function setFile(file) {
  const problem = validateFile(file);
  els.fileError.hidden = !problem;
  els.fileError.textContent = problem || "";
  selectedFile = problem ? null : file;
  els.fileCard.hidden = !selectedFile;
  els.dropzone.hidden = Boolean(selectedFile);
  if (selectedFile) {
    els.fileName.textContent = file.name;
    els.fileSize.textContent = formatSize(file.size);
  }
  els.fileInput.value = "";
  updateSubmit();
}

function updateSubmit() {
  els.submit.disabled = !selectedFile || controller !== null;
}

// ------------------------------------------------------------ progress UI
const STAGE_TO_STEP = {
  extracted: "extract",
  queued: "extract",
  analyzing: "plan",
  planning: "plan",
  mapping: "map",
  reducing: "map",
  writing: "write",
};
const STEP_ORDER = ["extract", "plan", "map", "write"];

function stepEl(name) {
  return els.steps.querySelector(`[data-step="${name}"]`);
}

function resetSteps() {
  STEP_ORDER.forEach((name) => {
    const li = stepEl(name);
    li.className = "";
    li.querySelector("small").textContent = "";
    const bar = li.querySelector(".bar");
    if (bar) { bar.hidden = true; bar.firstElementChild.style.width = "0"; }
  });
  stepEl("extract").classList.add("active");
  stepEl("extract").querySelector("small").textContent = "Uploading and validating…";
}

function activateStep(name, message) {
  const index = STEP_ORDER.indexOf(name);
  STEP_ORDER.forEach((other, i) => {
    const li = stepEl(other);
    if (i < index) {
      if (!li.classList.contains("skipped")) { li.classList.remove("active"); li.classList.add("done"); }
    } else if (i === index) {
      li.classList.remove("done");
      li.classList.add("active");
    }
  });
  if (message) stepEl(name).querySelector("small").textContent = message;
}

function handleProgress(event) {
  const step = STAGE_TO_STEP[event.stage];
  if (event.stage === "planning" && /single pass/i.test(event.message)) {
    const mapStep = stepEl("map");
    mapStep.classList.add("skipped");
    mapStep.querySelector("small").textContent = "Not needed: the document fits in one pass.";
  }
  if (event.stage === "done") {
    STEP_ORDER.forEach((name) => {
      const li = stepEl(name);
      li.classList.remove("active");
      if (!li.classList.contains("skipped")) li.classList.add("done");
    });
    return;
  }
  if (!step) return;
  activateStep(step, event.message);
  if (step === "map" && event.total) {
    const bar = stepEl("map").querySelector(".bar");
    bar.hidden = false;
    const done = event.stage === "mapping" ? event.current - 1 : event.total;
    bar.firstElementChild.style.width = `${Math.round((100 * done) / event.total)}%`;
  }
}

function failActiveStep() {
  const active = els.steps.querySelector("li.active");
  if (active) { active.classList.remove("active"); active.classList.add("failed"); }
}

// ------------------------------------------------------------ output states
function show(state) {
  els.empty.hidden = state !== "empty";
  els.error.hidden = state !== "error";
  els.result.hidden = state !== "result";
}

function showError(error, fallbackTitle = "Something went wrong") {
  els.errorTitle.textContent = ERROR_TITLES[error.code] || fallbackTitle;
  els.errorMessage.textContent = error.message || "Unknown error.";
  show("error");
  failActiveStep();
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function pageLabel(section) {
  return section.start_page === section.end_page ? `Page ${section.start_page}` : `Pages ${section.start_page}–${section.end_page}`;
}

function renderResult(data) {
  lastResult = data;
  $("result-file").textContent = data.document.filename;
  $("result-title").textContent = data.title;
  $("result-tldr").textContent = data.tldr;

  const totalSeconds = (data.stats.extraction_ms + data.stats.summarization_ms) / 1000;
  const chips = [
    ["Pages", data.document.pages],
    ["Words", data.document.words.toLocaleString("en-US")],
    ["Strategy", data.stats.strategy === "map-reduce" ? `map-reduce (${data.stats.chunks} sections)` : "single pass"],
    ["LLM calls", data.stats.llm_calls],
    ["Time", `${totalSeconds.toFixed(1)}s`],
    ["Model", data.stats.model],
  ];
  $("result-stats").replaceChildren(...chips.map(([label, value]) => {
    const li = el("li", `${label}: `);
    li.appendChild(el("strong", String(value)));
    return li;
  }));

  $("result-summary").replaceChildren(...data.summary.split(/\n\s*\n/).map((p) => el("p", p)));
  $("result-points").replaceChildren(...data.key_points.map((p) => el("li", p)));

  const sectionsBlock = $("sections-block");
  sectionsBlock.hidden = data.sections.length === 0;
  $("result-sections").replaceChildren(...data.sections.map((section) => {
    const details = el("details", undefined, "section");
    details.appendChild(el("summary", pageLabel(section)));
    const ul = el("ul");
    section.notes.forEach((note) => ul.appendChild(el("li", note)));
    details.appendChild(ul);
    return details;
  }));
  show("result");
}

function toMarkdown(data) {
  const lines = [
    `# ${data.title}`,
    "",
    `*Source: ${data.document.filename} (${data.document.pages} pages) — summarised locally with ${data.stats.model}*`,
    "",
    `**TL;DR:** ${data.tldr}`,
    "",
    "## Summary",
    "",
    data.summary,
    "",
    "## Key points",
    "",
    ...data.key_points.map((p, i) => `${i + 1}. ${p}`),
  ];
  if (data.sections.length) {
    lines.push("", "## Section-by-section notes");
    data.sections.forEach((s) => {
      lines.push("", `### ${pageLabel(s)}`, "", ...s.notes.map((n) => `- ${n}`));
    });
  }
  return lines.join("\n") + "\n";
}

// ------------------------------------------------------------ request
function setBusy(busy) {
  els.submit.classList.toggle("loading", busy);
  els.submit.querySelector(".btn__label").textContent = busy ? "Summarizing…" : "Summarize PDF";
  els.removeFile.disabled = busy;
  els.form.querySelectorAll('input[name="length"]').forEach((input) => { input.disabled = busy; });
  els.cancel.hidden = !busy;
  clearInterval(timer);
  if (busy) {
    const started = performance.now();
    timer = setInterval(() => { els.elapsed.textContent = `${((performance.now() - started) / 1000).toFixed(1)}s`; }, 100);
  }
}

async function summarize() {
  const problem = validateFile(selectedFile);
  if (problem) { els.fileError.textContent = problem; els.fileError.hidden = false; return; }

  const body = new FormData();
  body.append("file", selectedFile);
  body.append("length", els.form.querySelector('input[name="length"]:checked').value);

  controller = new AbortController();
  updateSubmit();
  setBusy(true);
  els.progressCard.hidden = false;
  resetSteps();
  show("empty");
  let finished = false;

  try {
    const response = await fetch("/api/summarize/stream", { method: "POST", body, signal: controller.signal });
    if (!response.ok) {
      let error = { code: "", message: `The server returned an error (${response.status}).` };
      try { error = (await response.json()).error || error; } catch { /* not JSON */ }
      showError(error, `Request failed (${response.status})`);
      if (error.code === "ollama_unavailable" || error.code === "model_not_found") checkHealth();
      return;
    }

    // Read newline-delimited JSON events as they arrive.
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.type === "progress") handleProgress(event);
        else if (event.type === "result") { finished = true; handleProgress({ stage: "done" }); renderResult(event.data); }
        else if (event.type === "error") { finished = true; showError(event.error); }
      }
    }
    if (!finished) showError({ code: "internal_error", message: "The connection closed before the summary was finished." });
  } catch (error) {
    if (controller.signal.aborted) {
      showError({ code: "", message: "You cancelled the request." }, "Cancelled");
    } else {
      showError({ code: "", message: "The FastAPI backend is not responding. Make sure it is running and refresh the page." }, "Cannot reach the server");
    }
  } finally {
    controller = null;
    setBusy(false);
    updateSubmit();
  }
}

async function checkHealth() {
  els.status.className = "status status--checking";
  els.statusText.textContent = "Checking Ollama…";
  try {
    const data = await (await fetch("/api/health")).json();
    const cls = { ok: "status--ok", degraded: "status--warn" }[data.status] || "status--error";
    els.status.className = `status ${cls}`;
    els.statusText.textContent = data.status === "ok" ? `${data.model} ready` : data.status === "degraded" ? "Model missing" : "Ollama offline";
    els.status.title = `${data.message} (click to re-check)`;
  } catch {
    els.status.className = "status status--error";
    els.statusText.textContent = "Server offline";
  }
}

async function loadConfig() {
  try {
    const data = await (await fetch("/api/config")).json();
    limits.maxFileMb = data.max_file_mb;
    limits.maxPages = data.max_pages;
    els.limitsHint.textContent = `PDF only · up to ${data.max_file_mb} MB and ${data.max_pages} pages · text-based (not scanned)`;
  } catch { /* keep defaults */ }
}

// ------------------------------------------------------------ events
els.fileInput.addEventListener("change", () => setFile(els.fileInput.files[0]));
els.removeFile.addEventListener("click", () => setFile(null) || (els.fileError.hidden = true));
["dragenter", "dragover"].forEach((type) => els.dropzone.addEventListener(type, (e) => {
  e.preventDefault();
  els.dropzone.classList.add("dragover");
}));
["dragleave", "drop"].forEach((type) => els.dropzone.addEventListener(type, (e) => {
  e.preventDefault();
  els.dropzone.classList.remove("dragover");
}));
els.dropzone.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  if (files.length > 1) {
    els.fileError.textContent = "Please drop a single PDF file.";
    els.fileError.hidden = false;
    return;
  }
  setFile(files[0]);
});
els.form.addEventListener("submit", (e) => { e.preventDefault(); if (!controller) summarize(); });
els.cancel.addEventListener("click", () => controller && controller.abort());
els.status.addEventListener("click", checkHealth);
els.copy.addEventListener("click", async () => {
  if (!lastResult) return;
  try { await navigator.clipboard.writeText(toMarkdown(lastResult)); els.copy.textContent = "Copied!"; }
  catch { els.copy.textContent = "Copy failed"; }
  setTimeout(() => { els.copy.textContent = "Copy"; }, 1500);
});
els.download.addEventListener("click", () => {
  if (!lastResult) return;
  const blob = new Blob([toMarkdown(lastResult)], { type: "text/markdown" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${lastResult.document.filename.replace(/\.pdf$/i, "")}-summary.md`;
  link.click();
  URL.revokeObjectURL(link.href);
});

loadConfig();
checkHealth();
