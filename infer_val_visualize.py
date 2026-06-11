#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/mnt/hdd12tb/code/thucnd/Model-S2O-with-resshift-main/experiments/multisar_opera_hls/2026-03-31-16-58-00/images/val",
    )
    parser.add_argument("--start", type=int, default=230)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--contact_cols", type=int, default=5)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--cell_size", type=int, default=256)
    parser.add_argument("--grid_pad", type=int, default=2)
    return parser.parse_args()


def read_bgr(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def resize_like(image: np.ndarray, height: int, width: int):
    if image.shape[0] == height and image.shape[1] == width:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def split_grid_tiles(image: np.ndarray, cell_size: int, grid_pad: int):
    h, w = image.shape[:2]
    ncols = (w - grid_pad) // (cell_size + grid_pad)
    nrows = (h - grid_pad) // (cell_size + grid_pad)
    tiles = []
    for row in range(nrows):
        for col in range(ncols):
            y = grid_pad + row * (cell_size + grid_pad)
            x = grid_pad + col * (cell_size + grid_pad)
            tile = image[y:y + cell_size, x:x + cell_size]
            if tile.shape[0] == cell_size and tile.shape[1] == cell_size:
                tiles.append(tile)
    return tiles, nrows, ncols


def split_progress_final_tiles(progress_image: np.ndarray, cell_size: int, grid_pad: int):
    h, w = progress_image.shape[:2]
    ncols = (w - grid_pad) // (cell_size + grid_pad)
    nrows = (h - grid_pad) // (cell_size + grid_pad)
    if ncols < 1 or nrows < 1:
        return [], nrows, ncols

    final_col = ncols - 1
    out_tiles = []
    for row in range(nrows):
        y = grid_pad + row * (cell_size + grid_pad)
        x = grid_pad + final_col * (cell_size + grid_pad)
        tile = progress_image[y:y + cell_size, x:x + cell_size]
        if tile.shape[0] == cell_size and tile.shape[1] == cell_size:
            out_tiles.append(tile)
    return out_tiles, nrows, ncols


def make_title_bar(text: str, width: int, height: int = 36):
    bar = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (10, int(height * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return bar


def add_label(image: np.ndarray, label: str):
    title = make_title_bar(label, image.shape[1])
    return np.vstack([title, image])


def diff_heatmap(a: np.ndarray, b: np.ndarray):
    diff = cv2.absdiff(a, b)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(diff_gray, cv2.COLORMAP_TURBO)
    mae = float(diff_gray.mean())
    cv2.putText(
        heat,
        f"MAE:{mae:.2f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return heat


def pad_to_height(image: np.ndarray, target_height: int):
    if image.shape[0] == target_height:
        return image
    if image.shape[0] > target_height:
        return image[:target_height, :, :]
    pad_h = target_height - image.shape[0]
    return cv2.copyMakeBorder(image, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(16, 16, 16))


def build_row_canvas(index: int, sample_id: int, lq: np.ndarray, progress_final: np.ndarray, gt: np.ndarray):
    height, width = progress_final.shape[:2]
    lq = resize_like(lq, height, width)
    gt = resize_like(gt, height, width)

    diff_pg = diff_heatmap(progress_final, gt)
    diff_pl = diff_heatmap(progress_final, lq)

    p1 = add_label(lq, f"LQ #{index}-{sample_id}")
    p2 = add_label(progress_final, "Output(final)")
    p3 = add_label(gt, "GT")
    p4 = add_label(diff_pg, "|Output-GT|")
    p5 = add_label(diff_pl, "|Output-LQ|")

    panel_height = max(p.shape[0] for p in [p1, p2, p3, p4, p5])
    panels = [pad_to_height(panel, panel_height) for panel in [p1, p2, p3, p4, p5]]

    spacer = np.full((panel_height, 12, 3), 10, dtype=np.uint8)
    row = panels[0]
    for panel in panels[1:]:
        row = np.hstack([row, spacer, panel])
    return row


def build_contact_sheet(rows, cols=5, gap=16):
    if not rows:
        raise ValueError("No rows to build contact sheet")

    row_h = max(row.shape[0] for row in rows)
    row_w = max(row.shape[1] for row in rows)
    normalized = []
    for row in rows:
        canvas = np.full((row_h, row_w, 3), 8, dtype=np.uint8)
        canvas[: row.shape[0], : row.shape[1], :] = row
        normalized.append(canvas)

    total = len(normalized)
    ncols = max(1, cols)
    nrows = (total + ncols - 1) // ncols

    sheet_h = nrows * row_h + (nrows - 1) * gap
    sheet_w = ncols * row_w + (ncols - 1) * gap
    sheet = np.full((sheet_h, sheet_w, 3), 0, dtype=np.uint8)

    for i, row in enumerate(normalized):
        rr = i // ncols
        cc = i % ncols
        y0 = rr * (row_h + gap)
        x0 = cc * (row_w + gap)
        sheet[y0:y0 + row_h, x0:x0 + row_w, :] = row

    return sheet


def main():
    args = parse_args()
    val_dir = Path(args.val_dir)
    output_dir = Path(args.output) if args.output else (val_dir / f"observe_{args.start}_{args.start + args.count - 1}")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_indices = list(range(args.start, args.start + args.count))
    rows = []
    missing = []
    total_samples = 0

    for idx in selected_indices:
        lq_path = val_dir / f"lq_SAR-{idx}.png"
        progress_path = val_dir / f"progress-{idx}.png"
        gt_path = val_dir / f"gt-{idx}.png"

        if not (lq_path.exists() and progress_path.exists() and gt_path.exists()):
            missing.append(idx)
            continue

        lq_grid = read_bgr(lq_path)
        progress_grid = read_bgr(progress_path)
        gt_grid = read_bgr(gt_path)

        lq_tiles, _, _ = split_grid_tiles(lq_grid, args.cell_size, args.grid_pad)
        gt_tiles, _, _ = split_grid_tiles(gt_grid, args.cell_size, args.grid_pad)
        out_tiles, prow, pcol = split_progress_final_tiles(progress_grid, args.cell_size, args.grid_pad)

        count = min(len(out_tiles), len(lq_tiles), len(gt_tiles))
        if count == 0:
            missing.append(idx)
            continue

        for sample_id in range(count):
            row = build_row_canvas(idx, sample_id, lq_tiles[sample_id], out_tiles[sample_id], gt_tiles[sample_id])
            rows.append(row)
            total_samples += 1
            single_path = output_dir / f"compare-{idx}-{sample_id}.jpg"
            cv2.imwrite(str(single_path), row, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])

        meta_text = np.full((60, 980, 3), 8, dtype=np.uint8)
        cv2.putText(meta_text, f"Index {idx}: progress grid={prow}x{pcol}, exported samples={count}",
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / f"meta-{idx}.jpg"), meta_text, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])

    if not rows:
        raise RuntimeError("No valid image triplets found for the selected range")

    sheet = build_contact_sheet(rows, cols=args.contact_cols, gap=16)
    sheet_path = output_dir / f"contact_sheet_{args.start}_{args.start + args.count - 1}.jpg"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])

    print(f"Saved {len(rows)} small comparisons ({total_samples} samples) to: {output_dir}")
    print(f"Contact sheet: {sheet_path}")
    if missing:
        print(f"Missing indices ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
