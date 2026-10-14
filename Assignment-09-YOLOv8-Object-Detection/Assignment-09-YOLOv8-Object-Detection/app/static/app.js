"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), overlay: $("overlay"), placeholder: $("webcam-placeholder"),
  startBtn: $("start-btn"), cameraSelect: $("camera-select"), snapshotBtn: $("snapshot-btn"),
  modelSelect: $("model-select"), modelDesc: $("model-desc"), conf: $("conf"), confValue: $("conf-value"),
  iou: $("iou"), iouValue: $("iou-value"), showLabels: $("show-labels"),
  fps: $("stat-fps"), inf: $("stat-inf"), rtt: $("stat-rtt"), count: $("stat-count"), deviceInfo: $("device-info"),
  detList: $("det-list"), error: $("error"), conn: $("conn"), connText: $("conn-text"),
  imageInput: $("image-input"), dropzone: $("dropzone"), imageStage: $("image-stage"), imageCanvas: $("image-canvas"),
};

const PALETTE = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4", "#ec4899", "#84cc16",
  "#f97316", "#14b8a6", "#6366f1", "#eab308"];
const colorFor = (classId) => PALETTE[classId % PALETTE.length];
const FRAME_WIDTH = 640; // frames are downscaled before sending: YOLO runs at 640 px anyway

let models = [];
let ws = null;
let wsReady = false;
let reconnectDelay = 500;
let stream = null;
let running = false;
let awaitingFrame = false;
let sentAt = 0;
let lastResult = null;
const frameTimes = [];
const captureCanvas = document.createElement("canvas");

// ------------------------------------------------------------------ helpers
function showError(message) {
  els.error.textContent = message;
  els.error.hidden = !message;
}

function settings() {
  return { type: "config", model: els.modelSelect.value, conf: parseFloat(els.conf.value), iou: parseFloat(els.iou.value) };
}

function drawDetections(ctx, result, scaleX, scaleY, offsetX = 0, offsetY = 0) {
  const showLabels = els.showLabels.checked;
  const lineWidth = Math.max(2, Math.round(ctx.canvas.width / 400));
  ctx.font = `600 ${Math.max(12, Math.round(ctx.canvas.width / 70))}px -apple-system, Segoe UI, sans-serif`;
  ctx.textBaseline = "top";
  for (const det of result.detections) {
    const [x1, y1, x2, y2] = det.box;
    const x = offsetX + x1 * scaleX;
    const y = offsetY + y1 * scaleY;
    const w = (x2 - x1) * scaleX;
    const h = (y2 - y1) * scaleY;
    const color = colorFor(det.class_id);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x, y, w, h);
    if (showLabels) {
      const text = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
      const padding = 4;
      const textWidth = ctx.measureText(text).width;
      const textHeight = parseInt(ctx.font.match(/(\d+)px/)[1], 10);
      const labelY = y - textHeight - padding * 2 >= 0 ? y - textHeight - padding * 2 : y;
      ctx.fillStyle = color;
      ctx.fillRect(x - lineWidth / 2, labelY, textWidth + padding * 2, textHeight + padding * 2);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, x + padding - lineWidth / 2, labelY + padding);
    }
  }
}

function renderList(result) {
  if (!result || !result.detections.length) {
    els.detList.innerHTML = '<li class="muted">No objects above the confidence threshold.</li>';
    return;
  }
  els.detList.replaceChildren(...result.detections.map((det) => {
    const li = document.createElement("li");
    li.className = "det";
    const swatch = document.createElement("span");
    swatch.className = "det__swatch";
    swatch.style.background = colorFor(det.class_id);
    const label = document.createElement("span");
    label.className = "det__label";
    label.textContent = det.label;
    const conf = document.createElement("span");
    conf.className = "det__conf";
    conf.textContent = `${(det.confidence * 100).toFixed(1)}%`;
    li.append(swatch, label, conf);
    return li;
  }));
}

// ------------------------------------------------------------------ WebSocket
function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${location.host}/ws/detect`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    wsReady = true;
    reconnectDelay = 500;
    els.conn.className = "status status--ok";
    els.connText.textContent = "Server connected";
    if (els.modelSelect.value) ws.send(JSON.stringify(settings()));
    awaitingFrame = false;
    if (running) requestAnimationFrame(sendFrame);
  };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "error") { showError(data.message); awaitingFrame = false; return; }
    if (data.type !== "detections") return;
    awaitingFrame = false;
    lastResult = data;
    const now = performance.now();
    frameTimes.push(now);
    while (frameTimes.length && now - frameTimes[0] > 1000) frameTimes.shift();
    els.fps.textContent = frameTimes.length;
    els.inf.textContent = `${data.inference_ms.toFixed(0)} ms`;
    els.rtt.textContent = `${(now - sentAt).toFixed(0)} ms`;
    els.count.textContent = data.detections.length;
    renderOverlay();
    renderList(data);
    if (running) requestAnimationFrame(sendFrame);
  };
  ws.onclose = () => {
    wsReady = false;
    els.conn.className = "status status--error";
    els.connText.textContent = "Disconnected – retrying…";
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 8000);
  };
}

// ------------------------------------------------------------------ webcam
async function listCameras() {
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    els.cameraSelect.hidden = devices.length < 2;
    const current = stream && stream.getVideoTracks()[0].getSettings().deviceId;
    els.cameraSelect.replaceChildren(...devices.map((d, i) => {
      const option = new Option(d.label || `Camera ${i + 1}`, d.deviceId);
      option.selected = d.deviceId === current;
      return option;
    }));
  } catch { /* ignore */ }
}

async function startCamera(deviceId) {
  showError("");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("This browser cannot access the camera. Open the app on http://localhost or over HTTPS in a modern browser.");
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId } } : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
  } catch (error) {
    const messages = {
      NotAllowedError: "Camera permission was denied. Allow camera access in your browser settings and try again.",
      NotFoundError: "No camera was found. Connect a webcam or use the 'Upload an image' tab.",
      NotReadableError: "The camera is already in use by another application.",
    };
    showError(messages[error.name] || `Could not start the camera: ${error.message}`);
    return;
  }
  els.video.srcObject = stream;
  await els.video.play();
  running = true;
  els.placeholder.hidden = true;
  els.startBtn.textContent = "Stop camera";
  els.startBtn.className = "btn btn--danger";
  els.snapshotBtn.disabled = false;
  listCameras();
  awaitingFrame = false;
  requestAnimationFrame(sendFrame);
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  els.video.srcObject = null;
  els.placeholder.hidden = false;
  els.startBtn.textContent = "Start camera";
  els.startBtn.className = "btn btn--primary";
  els.snapshotBtn.disabled = true;
  lastResult = null;
  renderOverlay();
  els.fps.textContent = els.inf.textContent = els.rtt.textContent = els.count.textContent = "–";
}

function sendFrame() {
  if (!running || !wsReady || awaitingFrame || !els.video.videoWidth) {
    if (running && !awaitingFrame) setTimeout(() => requestAnimationFrame(sendFrame), 50);
    return;
  }
  const scale = Math.min(1, FRAME_WIDTH / els.video.videoWidth);
  captureCanvas.width = Math.round(els.video.videoWidth * scale);
  captureCanvas.height = Math.round(els.video.videoHeight * scale);
  captureCanvas.getContext("2d").drawImage(els.video, 0, 0, captureCanvas.width, captureCanvas.height);
  awaitingFrame = true;
  captureCanvas.toBlob((blob) => {
    if (!blob || !wsReady) { awaitingFrame = false; return; }
    sentAt = performance.now();
    blob.arrayBuffer().then((buffer) => ws.send(buffer));
  }, "image/jpeg", 0.8);
}

/** Draw boxes over the video, accounting for object-fit: contain letterboxing. */
function renderOverlay() {
  const canvas = els.overlay;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!lastResult || !els.video.videoWidth) return;
  const videoAspect = els.video.videoWidth / els.video.videoHeight;
  let drawW = canvas.width;
  let drawH = canvas.width / videoAspect;
  if (drawH > canvas.height) { drawH = canvas.height; drawW = canvas.height * videoAspect; }
  const offsetX = (canvas.width - drawW) / 2;
  const offsetY = (canvas.height - drawH) / 2;
  drawDetections(ctx, lastResult, drawW / lastResult.width, drawH / lastResult.height, offsetX, offsetY);
}

function saveSnapshot() {
  const canvas = document.createElement("canvas");
  canvas.width = els.video.videoWidth;
  canvas.height = els.video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(els.video, 0, 0);
  if (lastResult) drawDetections(ctx, lastResult, canvas.width / lastResult.width, canvas.height / lastResult.height);
  const link = document.createElement("a");
  link.download = `detection-${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

// ------------------------------------------------------------------ image upload
let uploadedImage = null;
let uploadedResult = null;

async function detectImage(file) {
  showError("");
  if (!file) return;
  if (!file.type.startsWith("image/")) { showError(`"${file.name}" is not an image.`); return; }
  if (file.size > 10 * 1024 * 1024) { showError("The image is larger than 10 MB."); return; }
  const body = new FormData();
  body.append("file", file);
  const { model, conf, iou } = settings();
  body.append("model", model);
  body.append("conf", conf);
  body.append("iou", iou);
  const started = performance.now();
  try {
    const response = await fetch("/api/detect", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) { showError(data.error ? data.error.message : "Detection failed."); return; }
    uploadedImage = await createImageBitmap(file);
    uploadedResult = data;
    drawUploaded();
    renderList(data);
    els.inf.textContent = `${data.inference_ms.toFixed(0)} ms`;
    els.rtt.textContent = `${(performance.now() - started).toFixed(0)} ms`;
    els.count.textContent = data.detections.length;
    els.fps.textContent = "–";
  } catch {
    showError("Cannot reach the server.");
  }
}

function drawUploaded() {
  if (!uploadedImage || !uploadedResult) return;
  els.imageStage.hidden = false;
  const canvas = els.imageCanvas;
  canvas.width = uploadedImage.width;
  canvas.height = uploadedImage.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(uploadedImage, 0, 0);
  drawDetections(ctx, uploadedResult, canvas.width / uploadedResult.width, canvas.height / uploadedResult.height);
}

// ------------------------------------------------------------------ setup
async function loadModels() {
  try {
    const data = await (await fetch("/api/models")).json();
    models = data.models;
    els.modelSelect.replaceChildren(...models.map((m) => {
      const option = new Option(m.label + (m.available ? "" : " (not available)"), m.key);
      option.disabled = !m.available;
      option.selected = m.key === data.default;
      return option;
    }));
    els.deviceInfo.textContent = `Inference device: ${data.device}`;
    updateModelDesc();
  } catch {
    showError("Could not load the model list from the server.");
  }
}

function updateModelDesc() {
  const model = models.find((m) => m.key === els.modelSelect.value);
  els.modelDesc.textContent = model ? `Classes: ${model.classes.length <= 12 ? model.classes.join(", ") : `${model.classes.length} COCO classes`}` : "";
}

function pushSettings() {
  els.confValue.textContent = parseFloat(els.conf.value).toFixed(2);
  els.iouValue.textContent = parseFloat(els.iou.value).toFixed(2);
  if (wsReady) ws.send(JSON.stringify(settings()));
}

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => { t.classList.toggle("active", t === tab); t.setAttribute("aria-selected", t === tab); });
  $("panel-webcam").hidden = tab.dataset.tab !== "webcam";
  $("panel-upload").hidden = tab.dataset.tab !== "upload";
  if (tab.dataset.tab !== "webcam" && running) stopCamera();
}));
els.startBtn.addEventListener("click", () => (running ? stopCamera() : startCamera()));
els.cameraSelect.addEventListener("change", () => { stopCamera(); startCamera(els.cameraSelect.value); });
els.snapshotBtn.addEventListener("click", saveSnapshot);
els.modelSelect.addEventListener("change", () => { updateModelDesc(); pushSettings(); });
els.conf.addEventListener("input", pushSettings);
els.iou.addEventListener("input", pushSettings);
els.showLabels.addEventListener("change", () => { renderOverlay(); drawUploaded(); });
els.imageInput.addEventListener("change", () => detectImage(els.imageInput.files[0]));
["dragenter", "dragover"].forEach((t) => els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((t) => els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.remove("dragover"); }));
els.dropzone.addEventListener("drop", (e) => detectImage(e.dataTransfer.files[0]));
window.addEventListener("resize", renderOverlay);

loadModels().then(connect);
