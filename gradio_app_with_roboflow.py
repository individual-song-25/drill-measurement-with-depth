import os
import tempfile
import math
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from roboflow import Roboflow

BOX_HEIGHT_MM = 60.0
BOX_LENGTH_MM = 300.0
BOX_WIDTH_MM  = 190.0

rf      = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY", "uBQUwbuRddQGIP6cxHCo"))
project = rf.workspace("witeks-workspace").project("dekracoating")
model   = project.version(4).model

try:
    from depth_model import DepthEstimator
    estimator = DepthEstimator(model_size="small")
    print("Depth Anything V2 loaded.")
except Exception as e:
    print(f"[WARN] Depth model not available: {e}")
    estimator = None

def extract_four_corners(segmentation):
    if not segmentation or not isinstance(segmentation, list):
        return None
    if not isinstance(segmentation[0], list):
        return None
    pts = np.array(segmentation[0]).reshape(-1, 2).astype(np.float32)
    if len(pts) < 4:
        return None
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    for eps in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        epsilon = eps * cv2.arcLength(hull, True)
        approx  = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None

def sort_corners(pts):
    s  = pts.sum(axis=1)
    d  = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def estimate_scale_from_polygon(stand_seg, stand_bbox):
    corners = extract_four_corners(stand_seg)
    if corners is not None:
        tl, tr, br, bl = sort_corners(corners)
        avg_h = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        avg_v = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        if avg_h > 0 and avg_v > 0:
            return avg_h / BOX_LENGTH_MM, avg_v / BOX_WIDTH_MM, "polygon"
    _, _, bw, bh = [float(v) for v in stand_bbox]
    px_x = bw / BOX_LENGTH_MM if bw > 0 else None
    px_y = bh / BOX_WIDTH_MM  if bh > 0 else None
    return px_x, px_y, "bbox_fallback"

def fit_ellipse_minor_axis(segmentation):
    if not segmentation or not isinstance(segmentation, list):
        return None
    if not isinstance(segmentation[0], list):
        return None
    pts = np.array(segmentation[0]).reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return None
    try:
        return min(cv2.fitEllipse(pts)[1])
    except cv2.error:
        return None

def get_depth_in_region(depth_map, x, y, w, h):
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2 = min(depth_map.shape[1] - 1, int(x + w))
    y2 = min(depth_map.shape[0] - 1, int(y + h))
    region = depth_map[y1:y2, x1:x2]
    return float(np.median(region)) if region.size > 0 else 0.0

def cat_color(name):
    h = hash(name) & 0xFFFFFF
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)

def roboflow_pred_to_ann(pred, cat_id):
    """Convert a Roboflow JSON prediction to a minimal COCO annotation dict."""
    cx, cy = pred["x"], pred["y"]
    w,  h  = pred["width"], pred["height"]
    bbox   = [cx - w / 2, cy - h / 2, w, h]

    seg = None
    if "points" in pred:
        flat = []
        for pt in pred["points"]:
            flat += [pt["x"], pt["y"]]
        seg = [flat]
    else:
        x0, y0 = bbox[0], bbox[1]
        seg = [[x0, y0, x0 + w, y0, x0 + w, y0 + h, x0, y0 + h]]

    return {"bbox": bbox, "segmentation": seg, "category_id": cat_id,
            "class": pred["class"]}

def process_image(pil_image, use_depth):
    if pil_image is None:
        return None, []

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_image.save(tmp.name)
        tmp_path = tmp.name

    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    result      = model.predict(tmp_path, confidence=40).json()
    predictions = result.get("predictions", [])
    os.unlink(tmp_path)

    print("Detected classes:", [p.get("class") for p in predictions])

    stand_preds = [p for p in predictions if "black" in p.get("class", "").lower()]
    drill_preds = [p for p in predictions
                   if "black" not in p.get("class", "").lower()]

    if not stand_preds:
        vis = img_cv.copy()
        cv2.putText(vis, "No black stand detected!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), []

    stand_anns = [roboflow_pred_to_ann(p, 0) for p in stand_preds]
    drill_anns = [roboflow_pred_to_ann(p, i + 1) for i, p in enumerate(drill_preds)]

    depth_map = None
    if use_depth and estimator is not None:
        try:
            depth_map = estimator.estimate_depth(img_cv)
        except Exception as e:
            print(f"[WARN] Depth failed: {e}")

    def stand_center(ann):
        x, y, w, h = ann["bbox"]
        return x + w / 2, y + h / 2

    def drill_center(ann):
        x, y, w, h = ann["bbox"]
        return x + w / 2, y + h / 2

    rows = []
    for d_ann in drill_anns:
        dc = drill_center(d_ann)
        ref = min(stand_anns,
                  key=lambda s: math.hypot(dc[0] - stand_center(s)[0],
                                           dc[1] - stand_center(s)[1]))

        px_x, px_y, scale_method = estimate_scale_from_polygon(
            ref["segmentation"], ref["bbox"])

        _, _, dw, dh = [float(v) for v in d_ann["bbox"]]
        sx, sy = d_ann["bbox"][0], d_ann["bbox"][1]

        width_mm      = round(dw / px_x, 2) if px_x else None
        height_mm_raw = round(dh / px_y, 2) if px_y else None

        depth_ratio = 1.0
        if depth_map is not None:
            rx, ry, rw, rh = ref["bbox"]
            box_d   = get_depth_in_region(depth_map, rx, ry, rw, rh)
            drill_d = get_depth_in_region(depth_map, sx, sy, dw, dh)
            if drill_d > 0:
                depth_ratio = round(box_d / drill_d, 4)

        height_mm_corrected = (round(height_mm_raw * depth_ratio, 2)
                                if height_mm_raw is not None else None)

        minor_px = fit_ellipse_minor_axis(d_ann["segmentation"])
        if minor_px is not None and px_x:
            radius_mm     = round((minor_px / 2) / px_x, 2)
            radius_method = "ellipse"
        else:
            radius_mm     = round(width_mm / 2, 2) if width_mm else None
            radius_method = "bbox_fallback"

        rows.append({
            "type"                : d_ann["class"],
            "confidence"          : round(drill_preds[drill_anns.index(d_ann)]["confidence"], 2),
            "width_mm"            : width_mm,
            "height_mm_raw"       : height_mm_raw,
            "height_mm_corrected" : height_mm_corrected,
            "radius_mm"           : radius_mm,
            "radius_method"       : radius_method,
            "depth_ratio"         : depth_ratio,
            "scale_method"        : scale_method,
        })

    vis = img_cv.copy()

    if depth_map is not None:
        d8      = (depth_map * 255).astype(np.uint8)
        dcol    = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
        dcol    = cv2.resize(dcol, (vis.shape[1], vis.shape[0]))
        vis     = cv2.addWeighted(vis, 0.7, dcol, 0.3, 0)

    for s_ann in stand_anns:
        x, y, w, h = [int(float(v)) for v in s_ann["bbox"]]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        corners = extract_four_corners(s_ann["segmentation"])
        if corners is not None:
            for pt in corners.astype(int):
                cv2.circle(vis, tuple(pt), 8, (0, 0, 255), -1)
        cv2.putText(vis, "black-stand", (x, max(y - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    for d_ann, row in zip(drill_anns, rows):
        x, y, w, h = [int(float(v)) for v in d_ann["bbox"]]
        color = cat_color(row["type"])
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        seg = d_ann.get("segmentation")
        if seg and isinstance(seg[0], list):
            pts = np.array(seg[0]).reshape(-1, 2).astype(np.float32)
            if len(pts) >= 5:
                try:
                    ellipse = cv2.fitEllipse(pts)
                    cv2.ellipse(vis, ellipse, (0, 255, 0), 2)
                except cv2.error:
                    pass

        label = f"{row['type']}  r={row['radius_mm']}mm"
        cv2.putText(vis, label, (x, max(y - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        if row["height_mm_corrected"] is not None:
            cv2.putText(vis, f"h={row['height_mm_corrected']}mm",
                        (x + w + 4, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis_rgb, rows

def run(pil_image, use_depth):
    vis, rows = process_image(pil_image, use_depth)
    if vis is None:
        return None, []
    table = []
    for r in rows:
        table.append([
            r["type"],
            r["confidence"],
            r["radius_mm"],
            r["width_mm"],
            r["height_mm_raw"],
            r["height_mm_corrected"],
            r["depth_ratio"],
            r["radius_method"],
            r["scale_method"],
        ])
    return vis, table

HEADERS = [
    "Type", "Confidence", "Radius (mm)", "Width (mm)",
    "Height raw (mm)", "Height corrected (mm)",
    "Depth ratio", "Radius method", "Scale method",
]

with gr.Blocks(title="Drill Scanner – Dekracoating") as demo:
    gr.Markdown("## 🔩 Drill Scanner – Dekracoating")
    gr.Markdown(
        "Upload a photo that includes the **black reference stand**. "
        "The app detects drills, measures their dimensions in mm using the stand "
        "as a scale reference, and shows the annotated image."
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input  = gr.Image(type="pil", label="Upload drill photo")
            use_depth  = gr.Checkbox(
                label="Enable depth correction (Depth Anything V2)",
                value=estimator is not None,
                interactive=estimator is not None,
            )
            run_btn = gr.Button("Analyse", variant="primary")

        with gr.Column(scale=2):
            img_output = gr.Image(label="Annotated image", type="numpy")

    gr.Markdown("### Measurements")
    table_out = gr.Dataframe(headers=HEADERS, interactive=False)

    run_btn.click(fn=run, inputs=[img_input, use_depth],
                  outputs=[img_output, table_out])

if __name__ == "__main__":
    demo.launch()