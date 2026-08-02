# Vision Detect — In-Browser Object & Face Detection

A lightweight computer-vision web app that detects everyday objects (phones, keyboards, mice, bottles, laptops, cups, and more) and faces — entirely in the browser. No backend, no image uploads: all inference runs client-side.

## Features

- **Live webcam detection** — real-time bounding boxes over the camera feed
- **Image upload** — drag & drop or pick an image for one-shot analysis
- **Face detection** — dedicated face detector (MediaPipe BlazeFace), toggleable
- **Adjustable confidence threshold** — filter detections interactively
- **Structured output** — per-class counts and a one-click **Export JSON** with labels, confidences, and bounding boxes
- **Privacy-friendly** — images never leave the device

## Detectable classes

Faces, plus the **80 COCO classes** from the object model, including:

`cell phone, mouse, keyboard, laptop, bottle, cup, book, scissors, remote, tv, clock, backpack, chair, person, …`

> **Note:** items outside the COCO vocabulary (e.g. *pen*, or specific products/brands/SKUs) are not detected by the pretrained model. Supporting them requires a custom-trained detector — planned as a future phase (see roadmap).

## Tech stack

| Layer | Choice |
|---|---|
| Object detection | TensorFlow.js + COCO-SSD (runs in-browser, WebGL-accelerated) |
| Face detection | MediaPipe Tasks Vision (BlazeFace short-range) |
| Frontend | Vanilla HTML/CSS/JS — zero build step |
| Hosting | Vercel (static) |

## Run locally

Any static file server works:

```bash
npx serve .
```

Then open the printed URL. (Opening `index.html` directly via `file://` also works in most browsers, but webcam access requires `http(s)://` or `localhost`.)

## Deploy

The site is pure static files — deploy the folder to any static host:

```bash
vercel --prod
```

## Roadmap (future phases)

- Custom-trained model for out-of-vocabulary items (pens, specific products/brands)
- Batch processing of image datasets with aggregate reports
- Server-side API (FastAPI + YOLO) for higher accuracy and automation
- Video file analysis and per-frame analytics

## JSON output format

```json
{
  "source": "photo.jpg",
  "image_size": { "width": 1280, "height": 960 },
  "confidence_threshold": 0.5,
  "total_detections": 3,
  "counts_by_class": { "cell phone": 1, "keyboard": 1, "face": 1 },
  "detections": [
    { "label": "cell phone", "confidence": 0.91, "bbox_xywh": [412, 220, 180, 340] }
  ]
}
```
