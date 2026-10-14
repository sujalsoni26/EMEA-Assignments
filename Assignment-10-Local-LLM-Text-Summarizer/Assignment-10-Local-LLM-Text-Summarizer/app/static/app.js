"use strict";

const SAMPLES = [
  `Coral reefs cover less than one percent of the ocean floor, yet they support roughly a quarter of all known marine species. Reefs are built by tiny animals called coral polyps, which live in a close partnership with microscopic algae. The algae live inside the coral's tissue and produce food through photosynthesis, while the coral provides shelter and nutrients. This relationship is what gives many corals their bright colours.

When ocean water becomes too warm for an extended period, corals become stressed and expel the algae. Without them, the coral turns white, a process known as bleaching. Bleached corals are not dead, but they are starving, and if high temperatures continue for several weeks many of them die. Scientists have recorded four global mass bleaching events since 1998, with the most recent beginning in 2023 and affecting reefs in more than fifty countries.

Reefs also protect coastlines by absorbing wave energy during storms, and they support fishing and tourism industries worth billions of dollars every year. Researchers are testing ways to help reefs survive, including breeding heat-tolerant corals, restoring damaged areas with coral fragments grown in nurseries, and reducing local pressures such as pollution and overfishing. However, most experts agree that the long-term survival of coral reefs depends on rapidly reducing greenhouse gas emissions to limit ocean warming.`,

  `Hospitals are increasingly using artificial intelligence to help doctors read medical images such as X-rays, CT scans and mammograms. These systems are trained on large collections of labelled images and learn to highlight areas that may show signs of disease. In several studies, AI tools have matched or exceeded the accuracy of individual radiologists at detecting certain conditions, such as small lung nodules or early breast cancer.

Supporters say the technology could reduce waiting times and ease pressure on overworked staff, especially in regions where specialists are scarce. In one large screening programme, using AI as a second reader reduced the workload of radiologists by almost half while finding slightly more cancers than the traditional approach of having two human readers.

However, experts warn that AI systems can perform poorly when they are used on patients or equipment that differ from the data they were trained on. A model developed in one country may be less accurate in another, and errors can be hard to spot because the systems rarely explain their decisions. Regulators are therefore asking developers to test their tools on diverse populations and to monitor their performance after deployment. Most hospitals currently use AI to support, rather than replace, human specialists, and final decisions remain with qualified doctors.`,

  `Dear team,

From the first of next month we will move to a hybrid working model. Everyone will be expected in the office on Tuesdays and Thursdays, which we have chosen as our shared collaboration days. The remaining three days can be worked from home or from the office, depending on your preference and the needs of your projects.

To support this change, the third floor is being redesigned with more meeting rooms, quiet focus areas and bookable desks. Desk booking will be done through the new workplace app, which will be available on your laptop and phone from next week. Please book a desk at least one day in advance on collaboration days, because space will be limited while the renovation is completed.

Managers will meet with their teams over the next two weeks to agree on how meetings and core hours will work. If you have a medical condition, caring responsibilities or another reason that makes office attendance difficult, please speak to your manager or the HR team, and we will look for a suitable arrangement.

We will review the new model after three months using an anonymous staff survey, and we will adjust the approach based on your feedback.

Best regards,
The Operations Team`,
];

const els = {
  form: document.getElementById("form"),
  text: document.getElementById("text"),
  counter: document.getElementById("counter"),
  hint: document.getElementById("hint"),
  submit: document.getElementById("submit-btn"),
  clear: document.getElementById("clear-btn"),
  sample: document.getElementById("sample-btn"),
  copy: document.getElementById("copy-btn"),
  cancel: document.getElementById("cancel-btn"),
  status: document.getElementById("status"),
  statusText: document.getElementById("status-text"),
  empty: document.getElementById("empty"),
  loading: document.getElementById("loading"),
  elapsed: document.getElementById("elapsed"),
  error: document.getElementById("error"),
  errorTitle: document.getElementById("error-title"),
  errorMessage: document.getElementById("error-message"),
  result: document.getElementById("result"),
  summary: document.getElementById("summary"),
  stats: document.getElementById("stats"),
  minWords: document.getElementById("min-words"),
};

const limits = { minWords: 30, maxChars: 12000 };
const CLIENT_TIMEOUT_MS = 200_000;
let sampleIndex = 0;
let controller = null;
let timer = null;
let lastSummary = "";

const fmt = (n) => n.toLocaleString("en-US");
const wordCount = (text) => (text.trim() ? text.trim().split(/\s+/).length : 0);

// ---------------------------------------------------------------- validation
function validate(text) {
  const trimmed = text.trim();
  const words = wordCount(trimmed);
  if (!trimmed) return { ok: false, message: "" };
  if (trimmed.length > limits.maxChars) {
    return { ok: false, message: `Too long: remove ${fmt(trimmed.length - limits.maxChars)} characters.` };
  }
  if (words < limits.minWords) {
    return { ok: false, message: `Add at least ${limits.minWords - words} more word${limits.minWords - words === 1 ? "" : "s"}.` };
  }
  return { ok: true, message: "" };
}

function updateInputState() {
  const text = els.text.value;
  const chars = text.trim().length;
  els.counter.textContent = `${fmt(wordCount(text))} words · ${fmt(chars)} / ${fmt(limits.maxChars)} characters`;
  els.counter.classList.toggle("over", chars > limits.maxChars);
  const { ok, message } = validate(text);
  els.hint.textContent = message;
  els.text.classList.toggle("invalid", Boolean(message) && chars > limits.maxChars);
  els.submit.disabled = !ok || controller !== null;
}

// ---------------------------------------------------------------- output states
function show(state) {
  els.empty.hidden = state !== "empty";
  els.loading.hidden = state !== "loading";
  els.error.hidden = state !== "error";
  els.result.hidden = state !== "result";
  els.copy.hidden = state !== "result";
}

function showError(title, message) {
  els.errorTitle.textContent = title;
  els.errorMessage.textContent = message;
  show("error");
}

const ERROR_TITLES = {
  validation_error: "Please check your text",
  ollama_unavailable: "Ollama is not reachable",
  model_not_found: "Model not installed",
  ollama_timeout: "The model took too long",
  empty_response: "No summary was produced",
  internal_error: "Server error",
};

function renderSummary(data) {
  lastSummary = data.summary;
  els.summary.replaceChildren();
  if (data.format === "bullets") {
    const ul = document.createElement("ul");
    data.summary.split("\n").map((l) => l.replace(/^-\s*/, "").trim()).filter(Boolean).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      ul.appendChild(li);
    });
    els.summary.appendChild(ul);
  } else {
    const p = document.createElement("p");
    p.textContent = data.summary;
    els.summary.appendChild(p);
  }

  const reduction = data.input_words > 0 ? Math.max(0, Math.round(100 * (1 - data.summary_words / data.input_words))) : 0;
  const stats = [
    ["Words", `${fmt(data.input_words)} → ${fmt(data.summary_words)}`],
    ["Shorter by", `${reduction}%`],
    ["Time", `${(data.duration_ms / 1000).toFixed(1)}s`],
    ["Model", data.model],
  ];
  if (data.chunks > 1) stats.push(["Chunks", String(data.chunks)]);
  els.stats.replaceChildren(...stats.map(([label, value]) => {
    const li = document.createElement("li");
    li.innerHTML = `${label}: <strong></strong>`;
    li.querySelector("strong").textContent = value;
    return li;
  }));
  show("result");
}

// ---------------------------------------------------------------- loading
function setLoading(isLoading) {
  els.submit.classList.toggle("loading", isLoading);
  els.submit.querySelector(".btn__label").textContent = isLoading ? "Summarizing…" : "Summarize";
  els.text.readOnly = isLoading;
  els.clear.disabled = isLoading;
  els.sample.disabled = isLoading;
  clearInterval(timer);
  if (isLoading) {
    const started = performance.now();
    els.elapsed.textContent = "0.0";
    timer = setInterval(() => {
      els.elapsed.textContent = ((performance.now() - started) / 1000).toFixed(1);
    }, 100);
    show("loading");
  }
}

// ---------------------------------------------------------------- API calls
async function summarize() {
  const { ok, message } = validate(els.text.value);
  if (!ok) {
    showError(ERROR_TITLES.validation_error, message || "Please enter some text to summarize.");
    return;
  }

  const payload = {
    text: els.text.value,
    // Note: form.elements.length is the number of controls, so query the radios directly.
    length: els.form.querySelector('input[name="length"]:checked').value,
    format: els.form.querySelector('input[name="format"]:checked').value,
  };

  controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort("timeout"), CLIENT_TIMEOUT_MS);
  setLoading(true);
  updateInputState();

  try {
    const response = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    let data = null;
    try { data = await response.json(); } catch { /* non-JSON body */ }

    if (!response.ok) {
      const err = data && data.error ? data.error : null;
      showError(
        (err && ERROR_TITLES[err.code]) || `Request failed (${response.status})`,
        (err && err.message) || "The server returned an unexpected response.",
      );
      if (err && (err.code === "ollama_unavailable" || err.code === "model_not_found")) checkHealth();
      return;
    }
    renderSummary(data);
  } catch (error) {
    if (controller.signal.aborted) {
      if (controller.signal.reason === "timeout") {
        showError("Request timed out", "No response after 200 seconds. Try a shorter text.");
      } else {
        showError("Cancelled", "The request was cancelled.");
      }
    } else {
      showError("Cannot reach the server", "The FastAPI backend is not responding. Make sure it is running and refresh the page.");
    }
  } finally {
    clearTimeout(timeoutId);
    controller = null;
    setLoading(false);
    updateInputState();
  }
}

async function checkHealth() {
  els.status.className = "status status--checking";
  els.statusText.textContent = "Checking Ollama…";
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const cls = data.status === "ok" ? "status--ok" : data.status === "degraded" ? "status--warn" : "status--error";
    els.status.className = `status ${cls}`;
    els.statusText.textContent = data.status === "ok" ? `${data.model} ready` : data.status === "degraded" ? "Model missing" : "Ollama offline";
    els.status.title = `${data.message} (click to re-check)`;
  } catch {
    els.status.className = "status status--error";
    els.statusText.textContent = "Server offline";
    els.status.title = "The FastAPI backend is not responding.";
  }
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) return;
    const data = await response.json();
    limits.minWords = data.min_words;
    limits.maxChars = data.max_chars;
    els.minWords.textContent = data.min_words;
    els.text.maxLength = data.max_chars + 2000; // allow pasting a little over so we can explain the limit
  } catch { /* keep defaults */ }
  updateInputState();
}

// ---------------------------------------------------------------- events
els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!controller) summarize();
});
els.text.addEventListener("input", updateInputState);
els.text.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    if (!els.submit.disabled) summarize();
  }
});
els.clear.addEventListener("click", () => {
  els.text.value = "";
  lastSummary = "";
  updateInputState();
  show("empty");
  els.text.focus();
});
els.sample.addEventListener("click", () => {
  els.text.value = SAMPLES[sampleIndex % SAMPLES.length];
  sampleIndex += 1;
  updateInputState();
});
els.cancel.addEventListener("click", () => controller && controller.abort("user"));
els.copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(lastSummary);
    els.copy.textContent = "Copied!";
  } catch {
    els.copy.textContent = "Copy failed";
  }
  setTimeout(() => { els.copy.textContent = "Copy"; }, 1500);
});
els.status.addEventListener("click", checkHealth);

loadConfig();
checkHealth();
