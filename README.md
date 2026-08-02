# Vision Detect — Capture, Detect & Label

A browser-based computer-vision tool: **capture photos from your webcam**, let a pretrained model suggest detections, then **correct labels or draw your own boxes** for items the model doesn't know (a yoghurt box, a pen, a specific product) — and build up a labeled **training dataset**, all client-side. No backend, no uploads.

**Live app:** https://vision-detect-theta.vercel.app

## Workflow

1. **📷 Camera** → Start camera → live detections overlay the feed → **📸 Capture**
2. The captured photo opens in **labeling mode**, pre-seeded with the model's suggestions
3. Fix anything:
   - **Rename** a suggested label (e.g. `bottle` → `yoghurt drink`)
   - **Delete** wrong boxes (✕)
   - **Drag on the image** to draw a new box for anything the model missed, and type any label you want
4. **💾 Save to dataset** — stored locally in your browser (IndexedDB)
5. Repeat. When you have enough samples, **⬇️ Export ZIP (YOLO)**

The 🖼️ Upload tab does the same for existing image files.

## Export formats

- **Per-image JSON** — labels, confidences, bounding boxes, per-class counts
- **Dataset ZIP** — ready for custom model training:
  ```
  images/img_0001.jpg      ← captured photos
  labels/img_0001.txt      ← YOLO format (class cx cy w h, normalized)
  classes.txt              ← class index → name
  dataset.json             ← full metadata (COCO-style manifest)
  ```

## What the pretrained model suggests

Faces (MediaPipe BlazeFace) plus the **80 COCO classes** (cell phone, mouse, keyboard, laptop, bottle, cup, book, scissors, person, …). Anything outside that vocabulary is labeled manually — that's the point: the exported dataset is exactly what's needed to **train a custom detector** (e.g. YOLOv8 fine-tune) that recognizes your specific items directly. That training step is the natural next phase of this project.

## Tech stack

| Layer | Choice |
|---|---|
| Object detection | TensorFlow.js + COCO-SSD (in-browser, WebGL) |
| Face detection | MediaPipe Tasks Vision (BlazeFace) |
| Labeling & dataset | Canvas pointer interactions + IndexedDB + JSZip |
| Frontend | Vanilla HTML/CSS/JS — zero build step |
| Hosting | Vercel (static) |

## Run locally

```bash
npx serve .
```

Webcam access requires `http://localhost` or `https://`.

## Deploy

```bash
vercel --prod
```

## Roadmap

- Train a custom YOLO model from exported datasets (pens, yoghurt boxes, any product)
- Batch processing and aggregate reports
- Optional server-side API for automation
