import {
  FaceDetector,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MEDIAPIPE_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

const PALETTE = [
  "#3fb6ff", "#7ee787", "#ffa657", "#ff7b72", "#d2a8ff",
  "#79c0ff", "#f2cc60", "#56d4dd", "#ff9bce", "#a5d6ff",
];
const classColors = new Map();
function colorFor(label) {
  if (!classColors.has(label)) {
    classColors.set(label, PALETTE[classColors.size % PALETTE.length]);
  }
  return classColors.get(label);
}

const els = {
  status: document.getElementById("model-status"),
  statusText: document.getElementById("model-status-text"),
  tabCamera: document.getElementById("tab-camera"),
  tabUpload: document.getElementById("tab-upload"),
  stage: document.getElementById("stage"),
  placeholder: document.getElementById("placeholder"),
  placeholderText: document.getElementById("placeholder-text"),
  startBtn: document.getElementById("start-webcam-btn"),
  uploadLabel: document.getElementById("upload-label"),
  fileInput: document.getElementById("file-input"),
  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),
  captureBtn: document.getElementById("capture-btn"),
  dropHint: document.getElementById("drop-hint"),
  confSlider: document.getElementById("conf-slider"),
  confValue: document.getElementById("conf-value"),
  faceToggle: document.getElementById("face-toggle"),
  stopBtn: document.getElementById("stop-btn"),
  reviewHint: document.getElementById("review-hint"),
  resultsTitle: document.getElementById("results-title"),
  summary: document.getElementById("summary"),
  list: document.getElementById("detections-list"),
  exportBtn: document.getElementById("export-btn"),
  reviewActions: document.getElementById("review-actions"),
  saveBtn: document.getElementById("save-btn"),
  retakeBtn: document.getElementById("retake-btn"),
  datasetStats: document.getElementById("dataset-stats"),
  exportDatasetBtn: document.getElementById("export-dataset-btn"),
  clearDatasetBtn: document.getElementById("clear-dataset-btn"),
};

const ctx = els.canvas.getContext("2d");

const state = {
  mode: "camera", // "camera" | "upload"
  objectModel: null,
  faceDetector: null,
  faceMode: null,
  stream: null,
  running: false, // live camera loop
  liveDetections: [],
  // Review = a captured/uploaded still being labeled.
  // { image: CanvasImageSource, width, height, source, annotations: [...], selected }
  review: null,
  drawing: null, // in-progress box drag {x0, y0, x1, y1}
};

function threshold() {
  return Number(els.confSlider.value) / 100;
}

function setStatus(kind, text) {
  els.status.className = `model-status ${kind}`;
  els.statusText.textContent = text;
}

// ---------- Model loading ----------

async function loadModels() {
  try {
    const [objectModel, vision] = await Promise.all([
      cocoSsd.load(),
      FilesetResolver.forVisionTasks(MEDIAPIPE_WASM),
    ]);
    state.objectModel = objectModel;
    state.faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
      runningMode: "IMAGE",
      minDetectionConfidence: 0.4,
    });
    state.faceMode = "IMAGE";
    setStatus("ready", "Models ready");
    els.startBtn.disabled = false;
    els.placeholderText.textContent =
      "Ready. Start the camera, then capture a photo to label it.";
  } catch (err) {
    console.error("Model loading failed:", err);
    setStatus("error", "Model loading failed — check connection & reload");
  }
}

async function ensureFaceMode(mode) {
  if (state.faceMode !== mode) {
    await state.faceDetector.setOptions({ runningMode: mode });
    state.faceMode = mode;
  }
}

// ---------- Detection ----------

async function detectObjects(source) {
  const preds = await state.objectModel.detect(source, 40, 0.1);
  return preds.map((p) => ({
    label: p.class,
    score: p.score,
    bbox: p.bbox.map((v) => Math.round(v)),
    type: "object",
  }));
}

function mapFaces(result) {
  return (result?.detections ?? []).map((d) => ({
    label: "face",
    score: d.categories[0]?.score ?? 0,
    bbox: [
      Math.round(d.boundingBox.originX),
      Math.round(d.boundingBox.originY),
      Math.round(d.boundingBox.width),
      Math.round(d.boundingBox.height),
    ],
    type: "face",
  }));
}

async function detectFacesImage(source) {
  await ensureFaceMode("IMAGE");
  return mapFaces(state.faceDetector.detect(source));
}

async function detectFacesVideo(video, ts) {
  await ensureFaceMode("VIDEO");
  return mapFaces(state.faceDetector.detectForVideo(video, ts));
}

// ---------- Drawing ----------

function drawBoxes(detections, selectedIdx = -1) {
  const scale = Math.max(els.canvas.width / 640, 1);
  const lineW = 2 * scale;
  const fontSize = 13 * scale;
  ctx.font = `600 ${fontSize}px -apple-system, sans-serif`;
  ctx.textBaseline = "top";

  detections.forEach((d, i) => {
    const [x, y, w, h] = d.bbox;
    const color = colorFor(d.label || "unlabeled");
    const isSel = i === selectedIdx;
    ctx.strokeStyle = color;
    ctx.lineWidth = isSel ? lineW * 2 : lineW;
    ctx.setLineDash(isSel ? [8 * scale, 5 * scale] : []);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    const name = d.label || "unlabeled";
    const text = d.score != null ? `${name} ${(d.score * 100).toFixed(0)}%` : name;
    const tw = ctx.measureText(text).width;
    const th = fontSize * 1.35;
    const ty = y - th < 0 ? y : y - th;
    ctx.fillStyle = color;
    ctx.fillRect(x - lineW / 2, ty, tw + 10 * scale, th);
    ctx.fillStyle = "#0d1117";
    ctx.fillText(text, x + 4 * scale, ty + fontSize * 0.18);
  });
}

// ---------- Live camera mode ----------

function visibleLive() {
  const t = threshold();
  return state.liveDetections.filter(
    (d) => d.score >= t && (els.faceToggle.checked || d.type !== "face")
  );
}

async function startWebcam() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 } },
      audio: false,
    });
  } catch (err) {
    console.error("Webcam access failed:", err);
    els.placeholderText.textContent =
      "Could not access the camera. Check browser permissions and try again.";
    return;
  }

  els.video.srcObject = state.stream;
  await els.video.play();
  els.canvas.width = els.video.videoWidth;
  els.canvas.height = els.video.videoHeight;

  exitReview();
  els.placeholder.classList.add("hidden");
  els.canvas.classList.remove("hidden");
  els.captureBtn.classList.remove("hidden");
  els.stopBtn.classList.remove("hidden");
  state.running = true;

  let lastDetect = 0;
  const DETECT_EVERY_MS = 140;

  const loop = async (ts) => {
    if (!state.running) return;
    ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);

    if (ts - lastDetect >= DETECT_EVERY_MS) {
      lastDetect = ts;
      try {
        const [objects, faces] = await Promise.all([
          detectObjects(els.video),
          els.faceToggle.checked
            ? detectFacesVideo(els.video, ts)
            : Promise.resolve([]),
        ]);
        state.liveDetections = [...objects, ...faces];
      } catch (err) {
        console.error("Detection error:", err);
      }
    }

    const visible = visibleLive();
    drawBoxes(visible);
    renderLivePanel(visible);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function stopWebcam() {
  state.running = false;
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  els.captureBtn.classList.add("hidden");
  els.stopBtn.classList.add("hidden");
}

function renderLivePanel(detections) {
  els.exportBtn.disabled = detections.length === 0;
  if (detections.length === 0) {
    els.summary.innerHTML =
      '<p class="muted">Point the camera at objects — detections appear live. Hit 📸 Capture to freeze &amp; label.</p>';
    els.list.innerHTML = "";
    return;
  }
  renderChips(detections);
  els.list.innerHTML = detections
    .slice()
    .sort((a, b) => b.score - a.score)
    .map(
      (d) => `<li>
        <span class="label" style="color:${colorFor(d.label)}">● ${d.label}</span>
        <span class="score">${(d.score * 100).toFixed(1)}%</span>
      </li>`
    )
    .join("");
}

function renderChips(detections) {
  const counts = new Map();
  for (const d of detections) {
    const label = d.label || "unlabeled";
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  els.summary.innerHTML = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(
      ([label, n]) =>
        `<span class="chip" style="border-color:${colorFor(label)}">${label} <b>×${n}</b></span>`
    )
    .join("");
}

// ---------- Capture → review ----------

async function capturePhoto() {
  const frame = document.createElement("canvas");
  frame.width = els.video.videoWidth;
  frame.height = els.video.videoHeight;
  frame.getContext("2d").drawImage(els.video, 0, 0);
  stopWebcam();
  els.stage.classList.add("flash");
  setTimeout(() => els.stage.classList.remove("flash"), 400);
  await enterReview(frame, "webcam-capture");
}

async function enterReview(imageSource, sourceName) {
  const width = imageSource.naturalWidth ?? imageSource.width;
  const height = imageSource.naturalHeight ?? imageSource.height;
  els.canvas.width = width;
  els.canvas.height = height;
  els.placeholder.classList.add("hidden");
  els.canvas.classList.remove("hidden");
  els.canvas.classList.add("labeling");
  els.reviewActions.classList.remove("hidden");
  els.reviewHint.classList.remove("hidden");
  els.resultsTitle.textContent = "Labels";
  ctx.drawImage(imageSource, 0, 0);

  state.review = {
    image: imageSource,
    width,
    height,
    source: sourceName,
    annotations: [],
    selected: -1,
  };

  // Seed with model suggestions (filtered by current threshold).
  try {
    const [objects, faces] = await Promise.all([
      detectObjects(imageSource),
      els.faceToggle.checked ? detectFacesImage(imageSource) : Promise.resolve([]),
    ]);
    const t = threshold();
    state.review.annotations = [...objects, ...faces].filter((d) => d.score >= t);
  } catch (err) {
    console.error("Detection error:", err);
  }
  renderReview();
}

function exitReview() {
  state.review = null;
  state.drawing = null;
  els.canvas.classList.remove("labeling");
  els.reviewActions.classList.add("hidden");
  els.reviewHint.classList.add("hidden");
  els.resultsTitle.textContent = "Detections";
}

function renderReview() {
  const r = state.review;
  if (!r) return;
  ctx.drawImage(r.image, 0, 0);
  drawBoxes(r.annotations, r.selected);
  if (state.drawing) {
    const { x0, y0, x1, y1 } = state.drawing;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = Math.max(els.canvas.width / 640, 1) * 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    ctx.setLineDash([]);
  }
  renderLabelList();
  els.exportBtn.disabled = r.annotations.length === 0;
  if (r.annotations.length) renderChips(r.annotations);
  else
    els.summary.innerHTML =
      '<p class="muted">No labels yet — drag on the image to draw a box, then type a label.</p>';
}

function renderLabelList() {
  const r = state.review;
  els.list.innerHTML = "";
  r.annotations.forEach((a, i) => {
    const li = document.createElement("li");
    li.className = "clickable" + (i === r.selected ? " selected" : "");

    const dot = document.createElement("span");
    dot.textContent = "●";
    dot.style.color = colorFor(a.label || "unlabeled");

    const input = document.createElement("input");
    input.className = "label-input";
    input.value = a.label;
    input.placeholder = "type label…";
    input.addEventListener("input", () => {
      a.label = input.value.trim();
    });
    input.addEventListener("change", () => renderReview());
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") input.blur();
    });
    input.addEventListener("focus", () => {
      r.selected = i;
      // Redraw canvas only — rebuilding the list would kill focus.
      ctx.drawImage(r.image, 0, 0);
      drawBoxes(r.annotations, r.selected);
      [...els.list.children].forEach((el, j) =>
        el.classList.toggle("selected", j === i)
      );
    });

    const score = document.createElement("span");
    score.className = "score";
    score.textContent = a.score != null ? `${(a.score * 100).toFixed(0)}%` : "manual";

    const del = document.createElement("button");
    del.className = "del-btn";
    del.textContent = "✕";
    del.title = "Delete box";
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      r.annotations.splice(i, 1);
      r.selected = -1;
      renderReview();
    });

    li.addEventListener("click", () => {
      r.selected = i;
      renderReview();
    });

    li.append(dot, input, score, del);
    els.list.appendChild(li);
  });
}

// ---------- Canvas interaction (draw / select boxes) ----------

function canvasPoint(e) {
  // Map client coords → canvas pixel coords under object-fit: contain.
  const rect = els.canvas.getBoundingClientRect();
  const cw = els.canvas.width;
  const ch = els.canvas.height;
  const scale = Math.min(rect.width / cw, rect.height / ch);
  const offX = (rect.width - cw * scale) / 2;
  const offY = (rect.height - ch * scale) / 2;
  return {
    x: (e.clientX - rect.left - offX) / scale,
    y: (e.clientY - rect.top - offY) / scale,
  };
}

function hitTest(p) {
  const r = state.review;
  let best = -1;
  let bestArea = Infinity;
  r.annotations.forEach((a, i) => {
    const [x, y, w, h] = a.bbox;
    if (p.x >= x && p.x <= x + w && p.y >= y && p.y <= y + h && w * h < bestArea) {
      best = i;
      bestArea = w * h;
    }
  });
  return best;
}

els.canvas.addEventListener("pointerdown", (e) => {
  if (!state.review) return;
  e.preventDefault();
  try {
    els.canvas.setPointerCapture(e.pointerId);
  } catch {
    // Some browsers reject capture for non-standard pointers; drawing still works.
  }
  const p = canvasPoint(e);
  state.drawing = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
});

els.canvas.addEventListener("pointermove", (e) => {
  if (!state.review || !state.drawing) return;
  const p = canvasPoint(e);
  state.drawing.x1 = p.x;
  state.drawing.y1 = p.y;
  renderReviewCanvasOnly();
});

els.canvas.addEventListener("pointerup", (e) => {
  if (!state.review || !state.drawing) return;
  const r = state.review;
  const { x0, y0, x1, y1 } = state.drawing;
  state.drawing = null;
  const w = Math.abs(x1 - x0);
  const h = Math.abs(y1 - y0);
  const minSize = Math.max(els.canvas.width, els.canvas.height) * 0.012;

  if (w < minSize || h < minSize) {
    // Treat as a click: select box under cursor (or deselect).
    r.selected = hitTest(canvasPoint(e));
    renderReview();
    return;
  }

  const clamp = (v, max) => Math.max(0, Math.min(Math.round(v), max));
  r.annotations.push({
    label: "",
    score: null,
    bbox: [
      clamp(Math.min(x0, x1), r.width),
      clamp(Math.min(y0, y1), r.height),
      clamp(w, r.width),
      clamp(h, r.height),
    ],
    type: "manual",
  });
  r.selected = r.annotations.length - 1;
  renderReview();
  const inputs = els.list.querySelectorAll("input.label-input");
  inputs[inputs.length - 1]?.focus();
});

function renderReviewCanvasOnly() {
  const r = state.review;
  ctx.drawImage(r.image, 0, 0);
  drawBoxes(r.annotations, r.selected);
  if (state.drawing) {
    const { x0, y0, x1, y1 } = state.drawing;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = Math.max(els.canvas.width / 640, 1) * 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    ctx.setLineDash([]);
  }
}

// ---------- Dataset (IndexedDB) ----------

const DB_NAME = "vision-detect-dataset";
const STORE = "samples";

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbAdd(sample) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).add(sample);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbGetAll() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE).objectStore(STORE).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbClear() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function refreshDatasetStats() {
  try {
    const samples = await dbGetAll();
    const boxes = samples.reduce((n, s) => n + s.annotations.length, 0);
    const classes = new Set(
      samples.flatMap((s) => s.annotations.map((a) => a.label))
    );
    els.datasetStats.textContent = samples.length
      ? `${samples.length} images · ${boxes} boxes · ${classes.size} classes`
      : "empty";
    els.exportDatasetBtn.disabled = samples.length === 0;
    els.clearDatasetBtn.disabled = samples.length === 0;
  } catch (err) {
    console.error("Dataset stats failed:", err);
  }
}

async function saveToDataset() {
  const r = state.review;
  if (!r) return;
  const annotations = r.annotations.map((a) => ({
    label: a.label.trim() || "unlabeled",
    score: a.score,
    bbox: a.bbox,
  }));

  // Re-encode the still as JPEG (downscale very large images).
  const MAX_DIM = 1600;
  const scale = Math.min(1, MAX_DIM / Math.max(r.width, r.height));
  const out = document.createElement("canvas");
  out.width = Math.round(r.width * scale);
  out.height = Math.round(r.height * scale);
  out.getContext("2d").drawImage(r.image, 0, 0, out.width, out.height);
  const blob = await new Promise((res) => out.toBlob(res, "image/jpeg", 0.9));

  await dbAdd({
    created: new Date().toISOString(),
    source: r.source,
    width: out.width,
    height: out.height,
    annotations: annotations.map((a) => ({
      ...a,
      bbox: a.bbox.map((v) => Math.round(v * scale)),
    })),
    image: blob,
  });
  await refreshDatasetStats();

  els.saveBtn.textContent = "✅ Saved!";
  setTimeout(() => (els.saveBtn.textContent = "💾 Save to dataset"), 1200);
}

async function exportDatasetZip() {
  els.exportDatasetBtn.disabled = true;
  els.exportDatasetBtn.textContent = "Preparing…";
  try {
    const { default: JSZip } = await import(
      "https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm"
    );
    const samples = await dbGetAll();
    const classList = [
      ...new Set(samples.flatMap((s) => s.annotations.map((a) => a.label))),
    ].sort();
    const classIdx = new Map(classList.map((c, i) => [c, i]));

    const zip = new JSZip();
    zip.file("classes.txt", classList.join("\n") + "\n");

    const manifest = [];
    samples.forEach((s, i) => {
      const stem = `img_${String(i + 1).padStart(4, "0")}`;
      zip.file(`images/${stem}.jpg`, s.image);
      const yolo = s.annotations
        .map((a) => {
          const [x, y, w, h] = a.bbox;
          const cx = (x + w / 2) / s.width;
          const cy = (y + h / 2) / s.height;
          return `${classIdx.get(a.label)} ${cx.toFixed(6)} ${cy.toFixed(6)} ${(w / s.width).toFixed(6)} ${(h / s.height).toFixed(6)}`;
        })
        .join("\n");
      zip.file(`labels/${stem}.txt`, yolo + (yolo ? "\n" : ""));
      manifest.push({
        file: `images/${stem}.jpg`,
        source: s.source,
        created: s.created,
        width: s.width,
        height: s.height,
        annotations: s.annotations,
      });
    });
    zip.file("dataset.json", JSON.stringify({ classes: classList, samples: manifest }, null, 2));

    const blob = await zip.generateAsync({ type: "blob" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "vision-detect-dataset.zip";
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (err) {
    console.error("Dataset export failed:", err);
    alert("Export failed: " + err.message);
  } finally {
    els.exportDatasetBtn.textContent = "⬇️ Export ZIP (YOLO)";
    refreshDatasetStats();
  }
}

// ---------- Upload mode ----------

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const img = new Image();
  img.onload = async () => {
    URL.revokeObjectURL(img.src);
    await enterReview(img, file.name);
  };
  img.src = URL.createObjectURL(file);
}

// ---------- Mode switching ----------

function setMode(mode) {
  state.mode = mode;
  els.tabCamera.classList.toggle("active", mode === "camera");
  els.tabUpload.classList.toggle("active", mode === "upload");
  els.tabCamera.setAttribute("aria-selected", String(mode === "camera"));
  els.tabUpload.setAttribute("aria-selected", String(mode === "upload"));

  stopWebcam();
  exitReview();
  state.liveDetections = [];
  els.canvas.classList.add("hidden");
  els.placeholder.classList.remove("hidden");
  els.list.innerHTML = "";
  els.exportBtn.disabled = true;
  els.summary.innerHTML =
    '<p class="muted">Start the camera, capture a photo, then correct or add labels.</p>';

  const ready = Boolean(state.objectModel);
  if (mode === "camera") {
    els.startBtn.classList.remove("hidden");
    els.uploadLabel.classList.add("hidden");
    els.placeholderText.textContent = ready
      ? "Ready. Start the camera, then capture a photo to label it."
      : "Models are loading…";
  } else {
    els.startBtn.classList.add("hidden");
    els.uploadLabel.classList.remove("hidden");
    els.placeholderText.textContent = ready
      ? "Choose an image or drag & drop it here — then label it."
      : "Models are loading…";
  }
}

function retake() {
  if (state.mode === "camera") {
    startWebcam();
  } else {
    exitReview();
    els.canvas.classList.add("hidden");
    els.placeholder.classList.remove("hidden");
    els.list.innerHTML = "";
    els.summary.innerHTML =
      '<p class="muted">Choose another image to label.</p>';
    els.fileInput.value = "";
  }
}

// ---------- Per-image JSON export ----------

function exportJSON() {
  const detections = state.review ? state.review.annotations : visibleLive();
  const payload = {
    tool: "Vision Detect",
    source: state.review?.source ?? "webcam-live",
    exported_at: new Date().toISOString(),
    image_size: { width: els.canvas.width, height: els.canvas.height },
    total_detections: detections.length,
    counts_by_class: Object.fromEntries(
      detections.reduce(
        (m, d) => m.set(d.label || "unlabeled", (m.get(d.label || "unlabeled") ?? 0) + 1),
        new Map()
      )
    ),
    detections: detections.map((d) => ({
      label: d.label || "unlabeled",
      confidence: d.score != null ? Number(d.score.toFixed(4)) : null,
      bbox_xywh: d.bbox,
    })),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "detections.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------- Wiring ----------

els.tabCamera.addEventListener("click", () => setMode("camera"));
els.tabUpload.addEventListener("click", () => setMode("upload"));
els.startBtn.addEventListener("click", startWebcam);
els.stopBtn.addEventListener("click", () => {
  stopWebcam();
  els.canvas.classList.add("hidden");
  els.placeholder.classList.remove("hidden");
});
els.captureBtn.addEventListener("click", capturePhoto);
els.fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
els.exportBtn.addEventListener("click", exportJSON);
els.saveBtn.addEventListener("click", saveToDataset);
els.retakeBtn.addEventListener("click", retake);
els.exportDatasetBtn.addEventListener("click", exportDatasetZip);
els.clearDatasetBtn.addEventListener("click", async () => {
  if (confirm("Delete all saved images and labels from this browser?")) {
    await dbClear();
    await refreshDatasetStats();
  }
});

els.confSlider.addEventListener("input", () => {
  els.confValue.textContent = `${els.confSlider.value}%`;
});

// Drag & drop (upload mode)
["dragenter", "dragover"].forEach((evt) =>
  els.stage.addEventListener(evt, (e) => {
    e.preventDefault();
    if (state.mode !== "upload") return;
    els.stage.classList.add("dragover");
    els.dropHint.classList.remove("hidden");
  })
);
["dragleave", "drop"].forEach((evt) =>
  els.stage.addEventListener(evt, (e) => {
    e.preventDefault();
    els.stage.classList.remove("dragover");
    els.dropHint.classList.add("hidden");
    if (evt === "drop" && state.mode === "upload") {
      handleFile(e.dataTransfer.files[0]);
    }
  })
);

// Debug/automation hooks (harmless in production).
window.__visionDetect = {
  state,
  enterReviewFromFile: handleFile,
  dbGetAll,
  refreshDatasetStats,
};

setMode("camera");
loadModels();
refreshDatasetStats();
