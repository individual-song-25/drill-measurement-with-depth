# Drill Scanner – Dekracoating

A Gradio web application that detects industrial drills in photos, measures their real-world dimensions in millimetres, and displays an annotated image. Detection is powered by a Roboflow instance segmentation model. An optional depth correction step using Depth Anything V2 improves height accuracy when drills are at varying distances from the camera.

---

## How it works

1. The user uploads a photo containing drills and the black reference stand.
2. The Roboflow model detects and classifies all objects in the image.
3. The four corners of the black stand's segmentation polygon are extracted and used to calculate a pixel-to-millimetre scale (falls back to bounding box if the polygon cannot be reduced to four corners).
4. For each drill, OpenCV's `fitEllipse` is applied to the segmentation polygon. The minor axis of the fitted ellipse is used to estimate the drill radius, as it is the least affected by camera angle.
5. If depth correction is enabled, Depth Anything V2 generates a relative depth map. The median depth of each drill region is compared to the median depth of the black stand to produce a correction ratio that is applied to the raw height measurement.
6. The annotated image and a measurements table are shown in the interface.

---

## Requirements

- Python 3.9 or higher
- A Roboflow account with the Dekracoating project (version 4)

Install dependencies:

```bash
pip install gradio roboflow opencv-python pillow numpy
```

For depth correction, also install:

```bash
pip install torch transformers
```

---

## File structure

```
gradio_app_with_roboflow.py   # main application
depth_model.py                # Depth Anything V2 wrapper (optional)
```

---

## Configuration

The Roboflow API key is read from the environment variable `ROBOFLOW_API_KEY`. If the variable is not set, the key hardcoded in the file is used as a fallback.

To set the environment variable:

```bash
# macOS / Linux
export ROBOFLOW_API_KEY=your_key_here

# Windows
set ROBOFLOW_API_KEY=your_key_here
```

---

## Running the app

```bash
python gradio_app_with_roboflow.py
```

Then open `http://127.0.0.1:7860` in your browser.

---

## Usage

1. Upload a photo that includes the **black reference stand** and at least one drill.
2. Check or uncheck **Enable depth correction** (only available if `depth_model.py` is present).
3. Click **Analyse**.
4. The annotated image is shown on the right. The black stand is highlighted in cyan with red corner dots. Each drill gets a coloured bounding box, a green fitted ellipse, and labels showing its type, radius, and corrected height.
5. The measurements table below shows all detected drills with the following columns:

| Column | Description |
|--------|-------------|
| Type | Drill class name from Roboflow |
| Confidence | Detection confidence (0–1) |
| Radius (mm) | Estimated drill radius using ellipse minor axis |
| Width (mm) | Bounding box width converted to mm |
| Height raw (mm) | Bounding box height converted to mm |
| Height corrected (mm) | Height after depth correction |
| Depth ratio | Correction factor applied (1.0 = no correction) |
| Radius method | `ellipse` or `bbox_fallback` |
| Scale method | `polygon` or `bbox_fallback` |

---

## Black stand dimensions

The reference stand must always be visible in the photo. Its real-world dimensions are:

| Dimension | Value |
|-----------|-------|
| Length | 300 mm |
| Width | 190 mm |
| Height | 60 mm |

These values are hardcoded at the top of `gradio_app_with_roboflow.py` and must match the physical stand used during photography.

---

## Notes

- If the black stand is not detected, the app displays a warning and returns no measurements.
- Depth correction is relative, not absolute. It improves consistency across a single image but does not provide true distance values.
- For best accuracy, photograph drills from directly above with the black stand fully visible and unoccluded.
