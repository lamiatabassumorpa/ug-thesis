"""
Filter fake images to keep only those containing faces.
Uses OpenCV DNN face detector (fast, accurate).
This fixes content bias: ensures fake images are face images,
matching our real dataset (FFHQ + CelebA).

Usage:
  python scripts/filter_faces.py --input data/fake/diffusiondb_large --output data/fake/diffusiondb_faces
"""
import sys
import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL    = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
MODEL_DIR    = Path(__file__).resolve().parents[1] / "utils" / "face_detector"


def download_detector():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    proto = MODEL_DIR / "deploy.prototxt"
    model = MODEL_DIR / "res10_300x300_ssd.caffemodel"

    if not proto.exists():
        print("Downloading face detector prototxt...")
        urllib.request.urlretrieve(PROTOTXT_URL, proto)
    if not model.exists():
        print("Downloading face detector weights (~2 MB)...")
        urllib.request.urlretrieve(MODEL_URL, model)

    return str(proto), str(model)


def has_face(net, img_path: Path, conf_thresh: float = 0.5) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > conf_thresh:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',      required=True,  help='Dir with unfiltered fake images')
    parser.add_argument('--output',     required=True,  help='Dir for face-only fake images')
    parser.add_argument('--conf',       type=float, default=0.5)
    parser.add_argument('--max-images', type=int,   default=None)
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    proto, model = download_detector()
    net = cv2.dnn.readNetFromCaffe(proto, model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Face detector loaded.")

    images = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
    if args.max_images:
        images = images[:args.max_images]

    print(f"Filtering {len(images)} images from {input_dir}")

    kept = 0
    already = len(list(output_dir.glob("*.png")))
    print(f"Already have {already} face images in output dir.")

    for img_path in tqdm(images):
        out_path = output_dir / img_path.name
        if out_path.exists():
            kept += 1
            continue
        if has_face(net, img_path, args.conf):
            img_path.rename(out_path)
            kept += 1

    print(f"\nKept {kept}/{len(images)} images with faces → {output_dir}")
    print(f"Removed {len(images) - kept} non-face images.")


if __name__ == "__main__":
    main()
