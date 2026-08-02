import {
  FaceDetector,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MEDIAPIPE_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

// Distinct colors per class, assigned on first sight.
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
  tabWebcam: document.getElementById("tab-webcam"),
  tabUpload: document.getElementById("tab-upload"),
  stage: document.getElementById("stage"),
  placeholder: document.getElementById("placeholder"),
  placeholderText: document.getElementById("placeholder-text"),
  startBtn: document.getElementById("start-webcam-btn"),
  uploadLabel: document.getElementById("upload-label"),
  fileInput: document.getElementById("file-input"),
  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),
  dropHint: document.getElementById("drop-hint"),
  confSlider: document.getElementById("conf-slider"),
  confValue: document.getElementById("conf-value"),
  faceToggle: document.getElementById("face-toggle"),
  stopBtn: document.getElementById("stop-btn"),
  summary: document.getElementById("summary"),
  list: document.getElementById("detections-list"),
  exportBtn: document.getElementById("export-btn"),
};

const ctx = els.canvas.getContext("2d");

const state = {
  mode: "webcam", // "webcam" | "upload"
  objectModel: null,
  faceDetector: null,
  faceMode: null, // current MediaPipe running mode
  stream: null,
  running: false,
  rawDetections: [], // unfiltered, latest frame/image
  lastSource: null, // "webcam" | filename
  uploadedImage: null,
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
      "Ready. Start the webcam for live detection, or switch to the Upload tab.";
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
  // Low minScore here; the UI slider filters interactively.
  const preds = await state.objectModel.detect(source, 40, 0.1);
  return preds.map((p) => ({
    label: p.class,
    score: p.score,
    bbox: p.bbox.map((v) => Math.round(v)), // [x, y, w, h]
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

// ---------- Rendering ----------

function drawDetections(detections) {
  const scale = Math.max(els.canvas.width / 640, 1);
  const lineW = 2 * scale;
  const fontSize = 13 * scale;
  ctx.font = `600 ${fontSize}px -apple-system, sans-serif`;
  ctx.textBaseline = "top";

  for (const d of detections) {
    const [x, y, w, h] = d.bbox;
    const color = colorFor(d.label);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineW;
    ctx.strokeRect(x, y, w, h);

    const text = `${d.label} ${(d.score * 100).toFixed(0)}%`;
    const tw = ctx.measureText(text).width;
    const th = fontSize * 1.35;
    const ty = y - th < 0 ? y : y - th;
    ctx.fillStyle = color;
    ctx.fillRect(x - lineW / 2, ty, tw + 10 * scale, th);
    ctx.fillStyle = "#0d1117";
    ctx.fillText(text, x + 4 * scale, ty + fontSize * 0.18);
  }
}

function visibleDetections() {
  const t = threshold();
  return state.rawDetections.filter(
    (d) => d.score >= t && (els.faceToggle.checked || d.type !== "face")
  );
}

function renderResultsPanel(detections) {
  els.exportBtn.disabled = detections.length === 0;

  if (detections.length === 0) {
    els.summary.innerHTML =
      '<p class="muted">Nothing detected above the current threshold.</p>';
    els.list.innerHTML = "";
    return;
  }

  const counts = new Map();
  for (const d of detections) {
    counts.set(d.label, (counts.get(d.label) ?? 0) + 1);
  }
  els.summary.innerHTML = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(
      ([label, n]) =>
        `<span class="chip" style="border-color:${colorFor(label)}">${label} <b>×${n}</b></span>`
    )
    .join("");

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

function renderStill() {
  if (state.mode !== "upload" || !state.uploadedImage) return;
  ctx.drawImage(state.uploadedImage, 0, 0);
  const visible = visibleDetections();
  drawDetections(visible);
  renderResultsPanel(visible);
}

// ---------- Webcam mode ----------

async function startWebcam() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 } },
      audio: false,
    });
  } catch (err) {
    console.error("Webcam access failed:", err);
    els.placeholderText.textContent =
      "Could not access the webcam. Check browser permissions and try again.";
    return;
  }

  els.video.srcObject = state.stream;
  await els.video.play();
  els.canvas.width = els.video.videoWidth;
  els.canvas.height = els.video.videoHeight;

  els.placeholder.classList.add("hidden");
  els.canvas.classList.remove("hidden");
  els.stopBtn.classList.remove("hidden");
  state.running = true;
  state.lastSource = "webcam";

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
        state.rawDetections = [...objects, ...faces];
      } catch (err) {
        console.error("Detection error:", err);
      }
    }

    const visible = visibleDetections();
    drawDetections(visible);
    renderResultsPanel(visible);
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
  els.stopBtn.classList.add("hidden");
  els.canvas.classList.add("hidden");
  els.placeholder.classList.remove("hidden");
}

// ---------- Upload mode ----------

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const img = new Image();
  img.onload = async () => {
    URL.revokeObjectURL(img.src);
    state.uploadedImage = img;
    state.lastSource = file.name;
    els.canvas.width = img.naturalWidth;
    els.canvas.height = img.naturalHeight;
    els.placeholder.classList.add("hidden");
    els.canvas.classList.remove("hidden");
    ctx.drawImage(img, 0, 0);

    try {
      const [objects, faces] = await Promise.all([
        detectObjects(img),
        detectFacesImage(img),
      ]);
      state.rawDetections = [...objects, ...faces];
    } catch (err) {
      console.error("Detection error:", err);
      state.rawDetections = [];
    }
    renderStill();
  };
  img.src = URL.createObjectURL(file);
}

// ---------- Mode switching ----------

function setMode(mode) {
  state.mode = mode;
  els.tabWebcam.classList.toggle("active", mode === "webcam");
  els.tabUpload.classList.toggle("active", mode === "upload");
  els.tabWebcam.setAttribute("aria-selected", String(mode === "webcam"));
  els.tabUpload.setAttribute("aria-selected", String(mode === "upload"));

  stopWebcam();
  state.rawDetections = [];
  state.uploadedImage = null;
  els.canvas.classList.add("hidden");
  els.placeholder.classList.remove("hidden");
  renderResultsPanel([]);
  els.summary.innerHTML =
    '<p class="muted">No detections yet. Start the webcam or upload an image.</p>';

  const ready = Boolean(state.objectModel);
  if (mode === "webcam") {
    els.startBtn.classList.remove("hidden");
    els.uploadLabel.classList.add("hidden");
    els.placeholderText.textContent = ready
      ? "Ready. Start the webcam for live detection."
      : "Models are loading…";
  } else {
    els.startBtn.classList.add("hidden");
    els.uploadLabel.classList.remove("hidden");
    els.placeholderText.textContent = ready
      ? "Choose an image or drag & drop it here."
      : "Models are loading…";
  }
}

// ---------- Export ----------

function exportJSON() {
  const visible = visibleDetections();
  const payload = {
    tool: "Vision Detect",
    source: state.lastSource,
    exported_at: new Date().toISOString(),
    image_size: { width: els.canvas.width, height: els.canvas.height },
    confidence_threshold: threshold(),
    total_detections: visible.length,
    counts_by_class: Object.fromEntries(
      visible.reduce(
        (m, d) => m.set(d.label, (m.get(d.label) ?? 0) + 1),
        new Map()
      )
    ),
    detections: visible.map((d) => ({
      label: d.label,
      confidence: Number(d.score.toFixed(4)),
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

els.tabWebcam.addEventListener("click", () => setMode("webcam"));
els.tabUpload.addEventListener("click", () => setMode("upload"));
els.startBtn.addEventListener("click", startWebcam);
els.stopBtn.addEventListener("click", stopWebcam);
els.fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
els.exportBtn.addEventListener("click", exportJSON);

els.confSlider.addEventListener("input", () => {
  els.confValue.textContent = `${els.confSlider.value}%`;
  renderStill(); // webcam mode re-renders on its own loop
});
els.faceToggle.addEventListener("change", () => renderStill());

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

setMode("webcam");
loadModels();
