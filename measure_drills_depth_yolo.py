"""
measure_drills_yolo.py
======================
Measures the real-world height, width and radius of drill bits in images,
using a trained YOLO segmentation model for detection and classification,
and the black storage box as a pixel ruler.

Black box real-world dimensions (mm):
    HEIGHT  = 60  mm
    LENGTH  = 300 mm
    WIDTH   = 190 mm

Two modes:
    --mode yolo   : Run YOLO on new/unseen images (requires --model best.pt)
    --mode coco   : Use existing COCO JSON annotations (original behaviour)

Usage (YOLO mode - for new images):
    python measure_drills_yolo.py \
        --mode  yolo \
        --model best.pt \
        --images /path/to/images \
        --out   results.csv

Usage (COCO mode - for annotated dataset):
    python measure_drills_yolo.py \
        --mode coco \
        --coco _annotations.coco.json \
        --images /path/to/images \
        --out  results.csv

Add --depth to either mode to enable depth-based height correction.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ── Real-world black-box dimensions ──────────────────────────────────────────
BOX_HEIGHT_MM = 60.0
BOX_LENGTH_MM = 300.0
BOX_WIDTH_MM  = 190.0

STAND_CLASS = "black-stand"


# ── YOLO detection ────────────────────────────────────────────────────────────

def run_yolo(model, image, conf=0.25):
    """
    Run YOLO on a single image and return a list of annotation-style dicts,
    formatted exactly like COCO annotations so the rest of the pipeline
    doesn't need to change.

    Each dict has:
        category_id  : int (YOLO class index)
        bbox         : [x, y, w, h]  in pixels
        segmentation : [[x1,y1,x2,y2,...]]  polygon from mask
    """
    results = model(image, conf=conf, verbose=False)
    r = results[0]

    anns = []
    if r.boxes is None or len(r.boxes) == 0:
        return anns

    for i, box in enumerate(r.boxes):
        x_center, y_center, w, h = box.xywh[0].tolist()
        x = x_center - w / 2
        y = y_center - h / 2
        bbox = [x, y, w, h]

        # Get segmentation polygon from mask if available
        seg = None
        if r.masks is not None and i < len(r.masks.xy):
            mask_pts = r.masks.xy[i]
            if len(mask_pts) >= 5:
                seg = [mask_pts.flatten().tolist()]

        anns.append({
            "category_id" : int(box.cls),
            "bbox"        : bbox,
            "segmentation": seg,
            "conf"        : float(box.conf),
        })

    return anns


# ── Corner extraction ─────────────────────────────────────────────────────────

def extract_four_corners(segmentation):
    if not segmentation or not isinstance(segmentation, list):
        return None
    if not isinstance(segmentation[0], list):
        return None

    pts = np.array(segmentation[0]).reshape(-1, 2).astype(np.float32)
    if len(pts) < 4:
        return None

    hull = cv2.convexHull(pts.reshape(-1, 1, 2))

    for eps_factor in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        epsilon = eps_factor * cv2.arcLength(hull, True)
        approx  = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None


# ── Scale ─────────────────────────────────────────────────────────────────────

def estimate_scale_from_bbox(stand_bbox):
    _, _, bw, bh = [float(v) for v in stand_bbox]
    px_per_mm_x = bw / BOX_LENGTH_MM if bw > 0 else None
    px_per_mm_y = bh / BOX_WIDTH_MM  if bh > 0 else None
    return px_per_mm_x, px_per_mm_y


# ── Ellipse fitting ───────────────────────────────────────────────────────────

def fit_ellipse_minor_axis(segmentation):
    if not segmentation or not isinstance(segmentation, list):
        return None
    if not isinstance(segmentation[0], list):
        return None

    pts = np.array(segmentation[0]).reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return None

    try:
        ellipse = cv2.fitEllipse(pts)
        return min(ellipse[1])
    except cv2.error:
        return None


# ── Depth correction ──────────────────────────────────────────────────────────

def bbox_to_xyxy(bbox):
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def get_depth_ratio(depth_map, stand_bbox, drill_bbox, estimator):
    if depth_map is None or estimator is None:
        return 1.0

    box_depth   = estimator.get_depth_in_region(depth_map, bbox_to_xyxy(stand_bbox), method='median')
    drill_depth = estimator.get_depth_in_region(depth_map, bbox_to_xyxy(drill_bbox), method='median')

    if drill_depth == 0:
        return 1.0

    return box_depth / drill_depth


# ── Core measurement ──────────────────────────────────────────────────────────

def measure_drill(drill_ann, stand_ann, depth_map=None, estimator=None):
    stand_bbox = stand_ann["bbox"]
    drill_bbox = drill_ann["bbox"]
    drill_seg  = drill_ann.get("segmentation")

    px_per_mm_x, px_per_mm_y = estimate_scale_from_bbox(stand_bbox)
    _, _, dw, dh = [float(v) for v in drill_bbox]

    height_mm_raw = round(dh / px_per_mm_y, 2) if px_per_mm_y else None
    width_mm      = round(dw / px_per_mm_x, 2) if px_per_mm_x else None

    depth_ratio = get_depth_ratio(depth_map, stand_bbox, drill_bbox, estimator)

    height_mm_corrected = round(height_mm_raw * depth_ratio, 2) if height_mm_raw else None

    minor_px = fit_ellipse_minor_axis(drill_seg)

    if minor_px is not None and px_per_mm_x:
        radius_mm     = round((minor_px / 2) / px_per_mm_x, 2)
        radius_method = "ellipse"
    else:
        radius_mm     = round(width_mm / 2, 2) if width_mm else None
        radius_method = "bbox_fallback"

    return {
        "drill_px_width"     : round(dw, 1),
        "drill_px_height"    : round(dh, 1),
        "ellipse_minor_px"   : round(minor_px, 1) if minor_px else None,
        "height_mm_raw"      : height_mm_raw,
        "height_mm_corrected": height_mm_corrected,
        "depth_ratio"        : round(depth_ratio, 4),
        "width_mm"           : width_mm,
        "radius_mm"          : radius_mm,
        "radius_method"      : radius_method,
    }


# ── Pick closest stand ────────────────────────────────────────────────────────

def pick_reference_stand(stand_anns, drill_bbox):
    def center(bbox):
        x, y, w, h = [float(v) for v in bbox]
        return x + w / 2, y + h / 2

    dc = center(drill_bbox)
    best, best_dist = None, float("inf")
    for sa in stand_anns:
        sc = center(sa["bbox"])
        d  = math.hypot(dc[0] - sc[0], dc[1] - sc[1])
        if d < best_dist:
            best_dist = d
            best = sa
    return best


# ── Preview drawing ───────────────────────────────────────────────────────────

def cat_color(name):
    h = hash(name) & 0xFFFFFF
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)


def draw_annotations(image, annotations, categories, depth_map=None):
    vis = image.copy()

    if depth_map is not None:
        depth_uint8   = (depth_map * 255).astype(np.uint8)
        depth_color   = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        depth_resized = cv2.resize(depth_color, (vis.shape[1], vis.shape[0]))
        vis = cv2.addWeighted(vis, 0.7, depth_resized, 0.3, 0)

    for ann in annotations:
        name  = categories[ann["category_id"]]
        x, y, w, h = [int(float(v)) for v in ann["bbox"]]
        color = cat_color(name)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        seg = ann.get("segmentation")
        if seg and isinstance(seg, list) and isinstance(seg[0], list):
            if name == STAND_CLASS:
                corners = extract_four_corners(seg)
                if corners is not None:
                    for pt in corners.astype(int):
                        cv2.circle(vis, tuple(pt), 8, (0, 0, 255), -1)
            else:
                pts = np.array(seg[0]).reshape(-1, 2).astype(np.float32)
                if len(pts) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(pts)
                        cv2.ellipse(vis, ellipse, (0, 255, 0), 2)
                    except cv2.error:
                        pass

        # Show confidence if available (YOLO mode)
        label = name
        if "conf" in ann:
            label = f"{name} {ann['conf']:.0%}"
        cv2.putText(vis, label, (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


# ── COCO loading ──────────────────────────────────────────────────────────────

def load_coco(coco_path):
    with open(coco_path) as f:
        coco = json.load(f)

    categories  = {c["id"]: c["name"] for c in coco["categories"]}
    images_meta = {img["id"]: img      for img in coco["images"]}

    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    return categories, images_meta, img_anns


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode",       default="yolo", choices=["yolo", "coco"],
                        help="yolo = run YOLO on new images; coco = use existing annotations")
    parser.add_argument("--model",      default="best.pt",
                        help="Path to trained YOLO .pt file (yolo mode)")
    parser.add_argument("--coco",       help="Path to COCO JSON (coco mode)")
    parser.add_argument("--images",     required=True, help="Folder containing images")
    parser.add_argument("--out",        default="results.csv")
    parser.add_argument("--conf",       type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--depth",      action="store_true",
                        help="Enable depth correction using Depth Anything V2")
    parser.add_argument("--depth-size", default="small",
                        choices=["small", "base", "large"])
    args = parser.parse_args()

    # ── Load depth estimator ──────────────────────────────────────────────────
    estimator = None
    if args.depth:
        print(f"Loading Depth Anything V2 ({args.depth_size})...")
        try:
            from depth_model import DepthEstimator
            estimator = DepthEstimator(model_size=args.depth_size)
            print("Depth estimator ready.")
        except Exception as e:
            print(f"[WARN] Could not load depth estimator: {e}")

    # ── Load YOLO model ───────────────────────────────────────────────────────
    yolo_model = None
    if args.mode == "yolo":
        from ultralytics import YOLO
        print(f"Loading YOLO model: {args.model}")
        yolo_model  = YOLO(args.model)
        # Build category lookup from YOLO class names
        categories  = yolo_model.names   # {0: 'black-stand', 1: 'blauw-11', ...}
        print(f"YOLO ready. Classes: {list(categories.values())}")
    else:
        if not args.coco:
            parser.error("--coco is required when using --mode coco")
        categories, images_meta, img_anns = load_coco(args.coco)

    os.makedirs("annotated", exist_ok=True)
    results = []

    images_folder = Path(args.images)
    image_files   = sorted(
        list(images_folder.glob("*.jpg")) +
        list(images_folder.glob("*.jpeg")) +
        list(images_folder.glob("*.png"))
    )

    if args.mode == "yolo":
        # ── YOLO mode: iterate over image files ───────────────────────────────
        print(f"\nFound {len(image_files)} images in {images_folder}")

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Could not read {img_path.name}, skipping.")
                continue

            print(f"Processing {img_path.name}...")

            # Run YOLO
            anns = run_yolo(yolo_model, img, conf=args.conf)

            if not anns:
                print(f"  [WARN] No detections.")
                continue

            stand_anns = [a for a in anns if categories[a["category_id"]] == STAND_CLASS]
            drill_anns = [a for a in anns if categories[a["category_id"]] != STAND_CLASS]

            if not stand_anns:
                print(f"  [WARN] No black-stand detected — cannot measure.")
                continue

            print(f"  {len(drill_anns)} drills, {len(stand_anns)} stand(s) detected.")

            # Depth map
            depth_map = None
            if estimator:
                try:
                    depth_map = estimator.estimate_depth(img)
                except Exception as e:
                    print(f"  [WARN] Depth failed: {e}")

            for ann in drill_anns:
                ref_stand = pick_reference_stand(stand_anns, ann["bbox"])
                measures  = measure_drill(ann, ref_stand, depth_map, estimator)
                results.append({
                    "image_file" : img_path.name,
                    "drill_type" : categories[ann["category_id"]],
                    "conf"       : round(ann.get("conf", 0), 3),
                    **measures,
                })

            if not args.no_preview:
                vis = draw_annotations(img, anns, categories,
                                       depth_map=depth_map if args.depth else None)
                out_path = Path("annotated") / img_path.name
                cv2.imwrite(str(out_path), vis)

    else:
        # ── COCO mode: original behaviour ─────────────────────────────────────
        for image_id, anns in img_anns.items():
            img_info = images_meta.get(image_id)
            if img_info is None:
                continue

            stand_anns = [a for a in anns if categories[a["category_id"]] == STAND_CLASS]
            drill_anns = [a for a in anns if categories[a["category_id"]] != STAND_CLASS
                          and "-" in categories[a["category_id"]]]

            if not stand_anns:
                print(f"[WARN] Image {image_id}: no black-stand, skipping.")
                continue

            img_path = images_folder / img_info["file_name"]
            img = cv2.imread(str(img_path)) if img_path.exists() else None

            depth_map = None
            if estimator and img is not None:
                try:
                    depth_map = estimator.estimate_depth(img)
                except Exception as e:
                    print(f"[WARN] Depth failed for image {image_id}: {e}")

            for ann in drill_anns:
                ref_stand = pick_reference_stand(stand_anns, ann["bbox"])
                measures  = measure_drill(ann, ref_stand, depth_map, estimator)
                results.append({
                    "image_id"   : image_id,
                    "image_file" : img_info["file_name"],
                    "ann_id"     : ann["id"],
                    "drill_type" : categories[ann["category_id"]],
                    **measures,
                })

            if not args.no_preview and img is not None:
                vis = draw_annotations(img, anns, categories,
                                       depth_map=depth_map if args.depth else None)
                out_path = Path("annotated") / img_info["file_name"]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), vis)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if results:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} measurements -> {args.out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = defaultdict(list)
    for r in results:
        summary[r["drill_type"]].append(r)

    height_col = "height_mm_corrected" if estimator else "height_mm_raw"

    print(f"\n{'Drill type':<15} {'Count':>6} {'Avg height mm':>14} {'Avg width mm':>13} {'Avg radius mm':>14}")
    print("-" * 68)
    for dtype, rows in sorted(summary.items()):
        heights = [r[height_col] for r in rows if r[height_col]]
        widths  = [r["width_mm"]  for r in rows if r["width_mm"]]
        radii   = [r["radius_mm"] for r in rows if r["radius_mm"]]
        avg_h = sum(heights) / len(heights) if heights else 0
        avg_w = sum(widths)  / len(widths)  if widths  else 0
        avg_r = sum(radii)   / len(radii)   if radii   else 0
        print(f"{dtype:<15} {len(rows):>6} {avg_h:>14.1f} {avg_w:>13.1f} {avg_r:>14.1f}")


if __name__ == "__main__":
    main()
