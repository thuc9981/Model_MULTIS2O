import getpass
import json
import math
import os
import re
from datetime import timedelta
from pathlib import Path

import asf_search as asf
import dateutil.parser
import earthaccess
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window


PATCH_SIZE = 256
RAW_DIR = Path("/mnt/hdd1tb/SAR2Optical/Data_HLS_OPERA/raw")
CROP_DIR = Path("/mnt/hdd1tb/SAR2Optical/Data_HLS_OPERA/crop700")
MANIFEST = CROP_DIR / "paired_700_manifest.json"

S1_SEARCH_DAYS = 10
S2_SEARCH_DAYS = 10
S2_CLOUD = (0, 20)
DELTA_DEG = 0.08

S2_SHORTNAMES = [
    "SENTINEL-2A_MSI_L2A",
    "SENTINEL-2B_MSI_L2A",
    "SENTINEL-2A_MSI_L1C",
    "SENTINEL-2B_MSI_L1C",
]


def project_xy(x, y, src_crs, dst_crs):
    if str(src_crs) == str(dst_crs):
        return x, y
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return t.transform(x, y)


def center_window(src, cx, cy, size=PATCH_SIZE):
    row, col = src.index(cx, cy)
    row_off = int(round(row - size / 2))
    col_off = int(round(col - size / 2))
    if row_off < 0 or col_off < 0:
        return None
    if row_off + size > src.height or col_off + size > src.width:
        return None
    return Window(col_off=col_off, row_off=row_off, width=size, height=size)


def bad_ratio(masked_arr, nodata=None):
    if np.ma.isMaskedArray(masked_arr):
        m = np.ma.getmaskarray(masked_arr)
        if m.size:
            return float(m.mean())
    if nodata is None:
        return 0.0
    return float((masked_arr == nodata).mean())


def read_gray_patch(path, center_xy, center_crs, size=PATCH_SIZE):
    if path is None:
        return None, None
    with rasterio.open(path) as src:
        x, y = project_xy(center_xy[0], center_xy[1], center_crs, src.crs)
        win = center_window(src, x, y, size=size)
        if win is None:
            return None, None
        arr = src.read(1, window=win, masked=True).astype(np.float32)
        return arr, bad_ratio(arr, src.nodata)


def _hls_rgb_candidates_from_band(path):
    p = Path(path)
    m = re.search(r"\.B0?([0-9A-Za-z]+)\.tif[f]?$", p.name, flags=re.IGNORECASE)
    if not m:
        return None
    prefix = p.name[: m.start()]
    b04 = p.with_name(f"{prefix}.B04.tif")
    b03 = p.with_name(f"{prefix}.B03.tif")
    b02 = p.with_name(f"{prefix}.B02.tif")
    if b04.exists() and b03.exists() and b02.exists():
        return b04, b03, b02
    return None


def read_rgb_patch(path, center_xy, center_crs, size=PATCH_SIZE):
    if path is None:
        return None, None

    with rasterio.open(path) as src:
        x, y = project_xy(center_xy[0], center_xy[1], center_crs, src.crs)
        win = center_window(src, x, y, size=size)
        if win is None:
            return None, None

        if src.count >= 4:
            arr = src.read([4, 3, 2], window=win, masked=True).astype(np.float32)
            return arr, bad_ratio(arr, src.nodata)
        if src.count >= 3:
            arr = src.read([3, 2, 1], window=win, masked=True).astype(np.float32)
            return arr, bad_ratio(arr, src.nodata)

    # For HLS single-band files, build RGB from B04/B03/B02 companions
    cands = _hls_rgb_candidates_from_band(path)
    if cands is None:
        return None, None
    b04, b03, b02 = cands

    with rasterio.open(b04) as r4, rasterio.open(b03) as r3, rasterio.open(b02) as r2:
        x4, y4 = project_xy(center_xy[0], center_xy[1], center_crs, r4.crs)
        x3, y3 = project_xy(center_xy[0], center_xy[1], center_crs, r3.crs)
        x2, y2 = project_xy(center_xy[0], center_xy[1], center_crs, r2.crs)
        w4 = center_window(r4, x4, y4, size=size)
        w3 = center_window(r3, x3, y3, size=size)
        w2 = center_window(r2, x2, y2, size=size)
        if w4 is None or w3 is None or w2 is None:
            return None, None

        a4 = r4.read(1, window=w4, masked=True).astype(np.float32)
        a3 = r3.read(1, window=w3, masked=True).astype(np.float32)
        a2 = r2.read(1, window=w2, masked=True).astype(np.float32)
        arr = np.ma.stack([a4, a3, a2], axis=0)
        return arr, float(np.ma.getmaskarray(arr).mean())


def stretch_gray(arr):
    if arr is None:
        return None
    img = np.array(arr, dtype=np.float32)
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        img = (img - p2) / (p98 - p2)
    return np.clip(img, 0, 1)


def stretch_rgb(arr):
    if arr is None:
        return None
    rgb = np.moveaxis(np.array(arr, dtype=np.float32), 0, -1)
    p2, p98 = np.percentile(rgb, [2, 98])
    if p98 > p2:
        rgb = (rgb - p2) / (p98 - p2)
    return np.clip(rgb, 0, 1)


def normalize_paths(x):
    if x is None:
        return []
    if isinstance(x, (str, Path)):
        return [str(x)]
    if isinstance(x, (list, tuple, set)):
        return [str(p) for p in x if p is not None]
    return []


def pick_raster_path(paths):
    for ext in (".tif", ".tiff", ".jp2"):
        for p in normalize_paths(paths):
            if p.lower().endswith(ext):
                return p
    return None


def resolve_opera_tif(downloaded_paths, expected_name=None):
    p = pick_raster_path(downloaded_paths)
    if p:
        return p
    if expected_name:
        stem = Path(expected_name).stem
        found = list(RAW_DIR.glob(f"{stem}*.tif")) + list(RAW_DIR.glob(f"{stem}*.tiff")) + list(
            RAW_DIR.glob(f"{stem}*.jp2")
        )
        if found:
            return str(found[0])
    fallback = sorted(list(RAW_DIR.glob("OPERA*.tif")) + list(RAW_DIR.glob("OPERA*.tiff")))
    return str(fallback[-1]) if fallback else None


def build_bbox_wkt(lon, lat, delta=DELTA_DEG):
    minx, miny, maxx, maxy = lon - delta, lat - delta, lon + delta, lat + delta
    return f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"


def parse_hls_datetime(granule):
    temporal = granule.get("umm", {}).get("TemporalExtent", {})
    dt_str = None
    if isinstance(temporal.get("RangeDateTimes"), list) and temporal.get("RangeDateTimes"):
        dt_str = temporal["RangeDateTimes"][0].get("BeginningDateTime")
    elif isinstance(temporal.get("RangeDateTime"), dict):
        dt_str = temporal["RangeDateTime"].get("BeginningDateTime")
    elif isinstance(temporal.get("SingleDateTime"), str):
        dt_str = temporal.get("SingleDateTime")
    return dateutil.parser.isoparse(dt_str) if dt_str else None


def download_best_s1_for_center(lon, lat, ref_dt, asf_session):
    start = (ref_dt - timedelta(days=S1_SEARCH_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (ref_dt + timedelta(days=S1_SEARCH_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    wkt = build_bbox_wkt(lon, lat)

    results = asf.geo_search(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel="GRD_HD",
        intersectsWith=wkt,
        start=start,
        end=end,
    )
    if not results:
        results = asf.geo_search(
            platform=asf.PLATFORM.SENTINEL1,
            intersectsWith=wkt,
            start=start,
            end=end,
        )
    if not results:
        return None

    def _dt(prod):
        return dateutil.parser.isoparse(prod.properties["startTime"])

    ordered = sorted(results, key=lambda p: abs((_dt(p) - ref_dt).total_seconds()))
    for prod in ordered[:20]:
        files = normalize_paths(prod.download(path=str(RAW_DIR), session=asf_session))
        p = pick_raster_path(files)
        if p:
            return p
        expected = prod.properties.get("fileName") or prod.properties.get("fileID")
        if expected:
            stem = Path(expected).stem
            g = list(RAW_DIR.glob(f"{stem}*.tif")) + list(RAW_DIR.glob(f"{stem}*.tiff")) + list(
                RAW_DIR.glob(f"{stem}*.jp2")
            )
            if g:
                return str(g[0])
    return None


def download_best_s2_for_center(lon, lat, ref_dt):
    start = (ref_dt - timedelta(days=S2_SEARCH_DAYS)).strftime("%Y-%m-%d")
    end = (ref_dt + timedelta(days=S2_SEARCH_DAYS)).strftime("%Y-%m-%d")
    bbox = (lon - DELTA_DEG, lat - DELTA_DEG, lon + DELTA_DEG, lat + DELTA_DEG)

    best = None
    best_dt = None
    for short_name in S2_SHORTNAMES:
        try:
            results = earthaccess.search_data(
                short_name=short_name,
                bounding_box=bbox,
                temporal=(start, end),
                cloud_cover=S2_CLOUD,
            )
        except Exception:
            results = []

        for granule in results:
            dt = parse_hls_datetime(granule)
            if dt is None:
                continue
            if best is None or abs((dt - ref_dt).total_seconds()) < abs((best_dt - ref_dt).total_seconds()):
                best = granule
                best_dt = dt

    if best is None:
        return None

    files = normalize_paths(earthaccess.download([best], local_path=str(RAW_DIR)))
    return pick_raster_path(files)


def load_reference_pair():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    with open(MANIFEST, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    if not pairs:
        raise RuntimeError("Manifest has no pairs. Run the pipeline first.")
    ref = pairs[0]
    if not ref.get("hls_source") or not ref.get("opera_source"):
        raise RuntimeError("Manifest pair does not include hls_source/opera_source")
    return ref


def main():
    auth = earthaccess.login(strategy="interactive", persist=True)
    if auth is None:
        raise RuntimeError("Earthaccess login failed")

    token = os.getenv("EARTHDATA_TOKEN") or getpass.getpass("Earthdata token for ASF: ").strip()
    asf_session = asf.ASFSession()
    asf_session.auth_with_token(token)

    ref = load_reference_pair()
    lon, lat = ref["point"]
    hls_path = ref["hls_source"]
    opera_path = ref["opera_source"]
    hls_dt = dateutil.parser.isoparse(ref["hls_datetime"])
    opera_dt = dateutil.parser.isoparse(ref["opera_datetime"])

    with rasterio.open(hls_path) as hls_src:
        if ref.get("center_in_hls_crs") is not None:
            center_xy = (ref["center_in_hls_crs"][0], ref["center_in_hls_crs"][1])
        else:
            center_xy = project_xy(lon, lat, "EPSG:4326", hls_src.crs)
        center_crs = hls_src.crs

    print("Searching/downloading Sentinel-1 original ...")
    s1_path = download_best_s1_for_center(lon, lat, opera_dt, asf_session)
    print("Searching/downloading Sentinel-2 original ...")
    s2_path = download_best_s2_for_center(lon, lat, hls_dt)

    # 100% full patch requirement: bad ratio must be exactly 0 for all images
    s1_arr, s1_bad = read_gray_patch(s1_path, center_xy, center_crs)
    op_arr, op_bad = read_gray_patch(opera_path, center_xy, center_crs)
    s2_arr, s2_bad = read_rgb_patch(s2_path, center_xy, center_crs)
    hls_arr, hls_bad = read_rgb_patch(hls_path, center_xy, center_crs)

    errors = []
    if s1_arr is None or s1_bad > 0.0:
        errors.append(f"Sentinel-1 invalid/incomplete patch (bad={s1_bad})")
    if op_arr is None or op_bad > 0.0:
        errors.append(f"OPERA invalid/incomplete patch (bad={op_bad})")
    if s2_arr is None or s2_bad > 0.0:
        errors.append(f"Sentinel-2 RGB invalid/incomplete patch (bad={s2_bad})")
    if hls_arr is None or hls_bad > 0.0:
        errors.append(f"HLS RGB invalid/incomplete patch (bad={hls_bad})")

    if errors:
        raise RuntimeError("Could not build 100% full 256x256 comparison:\n- " + "\n- ".join(errors))

    img_s1 = stretch_gray(s1_arr)
    img_op = stretch_gray(op_arr)
    img_s2 = stretch_rgb(s2_arr)
    img_hls = stretch_rgb(hls_arr)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    ax = axes.ravel()

    ax[0].imshow(img_s1, cmap="gray")
    ax[0].set_title("Sentinel-1 Original (256x256)")
    ax[0].axis("off")

    ax[1].imshow(img_op, cmap="gray")
    ax[1].set_title("OPERA S1 (256x256)")
    ax[1].axis("off")

    ax[2].imshow(img_s2)
    ax[2].set_title("Sentinel-2 RGB (256x256)")
    ax[2].axis("off")

    ax[3].imshow(img_hls)
    ax[3].set_title("HLS RGB (256x256)")
    ax[3].axis("off")

    plt.tight_layout()
    plt.show()

    print("Comparison completed with strict 100% valid patches.")
    print(f"S1    : {s1_path}")
    print(f"OPERA : {opera_path}")
    print(f"S2    : {s2_path}")
    print(f"HLS   : {hls_path}")


if __name__ == "__main__":
    main()
