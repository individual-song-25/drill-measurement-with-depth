import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

BOX_HEIGHT_MM = 60.0
BOX_LENGTH_MM = 300.0
BOX_WIDTH_MM  = 190.0

PX_PER_MM     = 5.0
CANVAS_W      = int(BOX_LENGTH_MM * PX_PER_MM)
CANVAS_H      = int(BOX_WIDTH_MM  * PX_PER_MM)

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

def sort_corners(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def estimate_scale_from_bbox(stand_bbox):
    _, _, bw, bh = [float(v) for v in stand_bbox]
    px_per_mm_x = bw / BOX_LENGTH_MM if bw > 0 else None
    px_per_mm_y = bh / BOX_WIDTH_MM  if bh > 0 else None
    return px_per_mm_x, px_per_mm_y

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

def measure_drill(drill_ann, stand_ann, depth_map=None, estimator=None):
    stand_bbox = stand_ann["bbox"]
    drill_bbox = drill_ann["bbox"]
    drill_seg  = drill_ann.get("segmentation")

    px_per_mm_x, px_per_mm_y = estimate_scale_from_bbox(stand_bbox)
    _, _, dw, dh = [float(v) for v in drill_bbox]

    height_mm_raw = round(dh / px_per_mm_y, 2) if px_per_mm_y else None
    width_mm      = round(dw / px_per_mm_x, 2) if px_per_mm_x else None

    depth_ratio = get_depth_ratio(depth_map, stand_bbox, drill_bbox, estimator)
    
    if height_mm_raw is not None:
        height_mm_corrected = round(height_mm_raw * depth_ratio, 2)
    else:
        height_mm_corrected = None

    minor_px = fit_ellipse_minor_axis(drill_seg)

    if minor_px is not None and px_per_mm_x:
        radius_mm     = round((minor_px / 2) / px_per_mm_x, 2)
        radius_method = "ellipse"
    else:
        radius_mm     = round(width_mm / 2, 2) if width_mm else None
        radius_method = "bbox_fallback"

    return {
        "drill_px_width"    : round(dw, 1),
        "drill_px_height"   : round(dh, 1),
        "ellipse_minor_px"  : round(minor_px, 1) if minor_px else None,
        "height_mm_raw"     : height_mm_raw,
        "height_mm_corrected": height_mm_corrected,
        "depth_ratio"       : round(depth_ratio, 4),
        "width_mm"          : width_mm,
        "radius_mm"         : radius_mm,
        "radius_method"     : radius_method,
    }

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

def cat_color(name):
    h = hash(name) & 0xFFFFFF
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)

def draw_annotations(image, annotations, categories, depth_map=None):
    vis = image.copy()

    if depth_map is not None:
        depth_uint8  = (depth_map * 255).astype(np.uint8)
        depth_color  = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        depth_resized = cv2.resize(depth_color, (vis.shape[1], vis.shape[0]))
        vis = cv2.addWeighted(vis, 0.7, depth_resized, 0.3, 0)

    for ann in annotations:
        name  = categories[ann["category_id"]]
        x, y, w, h = [int(float(v)) for v in ann["bbox"]]
        color = cat_color(name)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        seg = ann.get("segmentation")
        if name == "black-stand" and seg and isinstance(seg[0], list):
            corners = extract_four_corners(seg)
            if corners is not None:
                for pt in corners.astype(int):
                    cv2.circle(vis, tuple(pt), 8, (0, 0, 255), -1)

        elif name != "black-stand" and seg and isinstance(seg, list) and isinstance(seg[0], list):
            pts = np.array(seg[0]).reshape(-1, 2).astype(np.float32)
            if len(pts) >= 5:
                try:
                    ellipse = cv2.fitEllipse(pts)
                    cv2.ellipse(vis, ellipse, (0, 255, 0), 2)
                except cv2.error:
                    pass

        cv2.putText(vis, name, (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco",        required=True,  help="Path to COCO JSON")
    parser.add_argument("--images",      required=True,  help="Folder containing images")
    parser.add_argument("--out",         default="results.csv")
    parser.add_argument("--no-preview",  action="store_true")
    parser.add_argument("--depth",       action="store_true",
                        help="Enable depth-based height correction using Depth Anything V2")
    parser.add_argument("--depth-size",  default="small", choices=["small", "base", "large"],
                        help="Depth Anything V2 model size (default: small)")
    args = parser.parse_args()

    estimator = None
    if args.depth:
        print(f"Loading Depth Anything V2 ({args.depth_size})...")
        try:
            from depth_model import DepthEstimator
            estimator = DepthEstimator(model_size=args.depth_size)
            print("Depth estimator ready.")
        except Exception as e:
            print(f"[WARN] Could not load depth estimator: {e}")
            print("[WARN] Continuing without depth correction.")

    with open(args.coco) as f:
        coco = json.load(f)

    categories  = {c["id"]: c["name"] for c in coco["categories"]}
    images_meta = {img["id"]: img     for img in coco["images"]}

    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    os.makedirs("annotated", exist_ok=True)

    results = []

    for image_id, anns in img_anns.items():
        img_info = images_meta.get(image_id)
        if img_info is None:
            continue

        stand_anns = [a for a in anns if categories[a["category_id"]] == "black-stand"]
        drill_anns = [a for a in anns if categories[a["category_id"]] != "black-stand"
                      and "-" in categories[a["category_id"]]]

        if not stand_anns:
            print(f"[WARN] Image {image_id}: no black-stand found, skipping.")
            continue

        img_path = Path(args.images) / img_info["file_name"]
        img = None
        if img_path.exists():
            img = cv2.imread(str(img_path))

        depth_map = None
        if estimator is not None and img is not None:
            try:
                depth_map = estimator.estimate_depth(img)
                print(f"  Depth map computed for image {image_id}")
            except Exception as e:
                print(f"[WARN] Depth estimation failed for image {image_id}: {e}")

        for ann in drill_anns:
            ref_stand = pick_reference_stand(stand_anns, ann["bbox"])
            measures  = measure_drill(ann, ref_stand, depth_map=depth_map, estimator=estimator)

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

    if results:
        fieldnames = list(results[0].keys())
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} measurements -> {args.out}")

    summary = defaultdict(list)
    for r in results:
        summary[r["drill_type"]].append(r)

    depth_active = estimator is not None
    height_col   = "height_mm_corrected" if depth_active else "height_mm_raw"

    print(f"\nHeight column used: {height_col}")
    print(f"\n{'Drill type':<15} {'Count':>6} {'Avg height mm':>14} {'Avg width mm':>13} {'Avg radius mm':>14} {'Avg depth ratio':>16}")
    print("-" * 85)
    for dtype, rows in sorted(summary.items()):
        heights = [r[height_col] for r in rows if r[height_col]]
        widths  = [r["width_mm"]  for r in rows if r["width_mm"]]
        radii   = [r["radius_mm"] for r in rows if r["radius_mm"]]
        ratios  = [r["depth_ratio"] for r in rows if r["depth_ratio"]]
        avg_h = sum(heights) / len(heights) if heights else 0
        avg_w = sum(widths)  / len(widths)  if widths  else 0
        avg_r = sum(radii)   / len(radii)   if radii   else 0
        avg_d = sum(ratios)  / len(ratios)  if ratios  else 1.0
        print(f"{dtype:<15} {len(rows):>6} {avg_h:>14.1f} {avg_w:>13.1f} {avg_r:>14.1f} {avg_d:>16.3f}")

if __name__ == "__main__":
    main()