"""
YOLO image-segmentation for microscope images — importable API with patch batching.

You can control RAM/GPU usage via `max_patches_in_memory`:
- 1  -> very low memory usage (serial patches, minimal RAM)
- >1 -> batches of patches are processed together (higher throughput, more RAM)

External usage:
---------------
from inference import run_inference, load_model, run_inference_with_model

paths = run_inference(
    image_path="data/.../sample.bmp",
    xml_path="data/.../info.xml",
    model_path="models/IFM_segmentation.pt",
    output_path="output",
    max_patches_in_memory=8,  # NEW
)
"""

from __future__ import annotations
from typing import Dict, Mapping, Optional, List, Tuple

from ultralytics import YOLO
import cv2
import numpy as np
import os
from utils.results_to_csv import save_segmentation_results
from utils.results_to_JSON import save_segmentation_results_json


# Defaults
DEFAULT_PATCH_SIZE = 640
DEFAULT_CLASS_COLORS: Mapping[int, tuple[int, int, int]] = {
    0: (0,   0,   255),   # class 0 → red
    1: (0,   255, 0  ),   # class 1 → green
    2: (255, 0,   0  ),   # class 2 → blue
}
DEFAULT_ALPHA = 0.7  # 0.0 = transparent, 1.0 = opaque


def load_model(model_path: str) -> YOLO:
    """Load a YOLO segmentation model from disk."""
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_path must be a non-empty string.")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(model_path)


def run_inference(
    image_path: str,
    xml_path: str,
    *,
    model_path: str = "models/IFM_segmentation.pt",
    output_path: str = "output",
    patch_size: int = DEFAULT_PATCH_SIZE,
    class_colors: Optional[Mapping[int, tuple[int, int, int]]] = None,
    alpha: float = DEFAULT_ALPHA,
    conf: float = 0.4,
    iou: float = 0.3,
    max_patches_in_memory: int = 8,   
) -> Dict[str, str]:
    """Convenience wrapper that loads the model internally."""
    model = load_model(model_path)
    return run_inference_with_model(
        model=model,
        image_path=image_path,
        xml_path=xml_path,
        output_path=output_path,
        patch_size=patch_size,
        class_colors=class_colors,
        alpha=alpha,
        conf=conf,
        iou=iou,
        max_patches_in_memory=max_patches_in_memory,  
    )


def run_inference_with_model(
    *,
    model: YOLO,
    image_path: str,
    xml_path: str,
    output_path: str = "output",
    patch_size: int = DEFAULT_PATCH_SIZE,
    class_colors: Optional[Mapping[int, tuple[int, int, int]]] = None,
    alpha: float = DEFAULT_ALPHA,
    conf: float = 0.4,
    iou: float = 0.3,
    max_patches_in_memory: int = 8,  
) -> Dict[str, str]:
    """
    Run segmentation inference with an already-loaded YOLO model, processing at most
    `max_patches_in_memory` patches concurrently to limit RAM usage.
    """
    # --- validation ---
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("image_path must be a non-empty string.")
    if not isinstance(xml_path, str) or not xml_path:
        raise ValueError("xml_path must be a non-empty string.")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0.0, 1.0].")
    if max_patches_in_memory < 1:
        raise ValueError("max_patches_in_memory must be >= 1.")
    if class_colors is None:
        class_colors = DEFAULT_CLASS_COLORS

    # Load image (note: this still holds the full image in RAM; for truly huge images consider a tiled format)
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Failed to load image: {image_path}")

    height, width = original.shape[:2]
    annotated = original.copy()
    all_results = []

    # Precompute patch coordinates
    coords: List[Tuple[int, int, int, int]] = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            coords.append((x, y, x_end, y_end))

    def batches(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    # Process patches in batches to limit peak memory
    for coord_batch in batches(coords, max_patches_in_memory):
        batch_inputs: List[np.ndarray] = []
        batch_meta:   List[Tuple[int,int,int,int,int,int]] = []  # (x, y, x_end, y_end, ph, pw)

        # 1) extract & pad current batch
        for (x, y, x_end, y_end) in coord_batch:
            patch = original[y:y_end, x:x_end]
            ph, pw = patch.shape[:2]
            if ph < patch_size or pw < patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                padded[:ph, :pw] = patch
            else:
                padded = patch
            batch_inputs.append(padded)
            batch_meta.append((x, y, x_end, y_end, ph, pw))

        # 2) inference on this batch
        results_batch = model.predict(
            batch_inputs,
            imgsz=patch_size,
            augment=False,
            conf=conf,
            iou=iou,
            stream=False,
            verbose=False,
        )

        # 3) post-process and reassemble
        for i, (result, meta) in enumerate(zip(results_batch, batch_meta)):
            x, y, x_end, y_end, ph, pw = meta
            padded = batch_inputs[i]  

            # absolute boxes & centroids 
            boxes = result.boxes.xyxy
            boxes_np = boxes.cpu().numpy() if hasattr(boxes, "cpu") else boxes
            if boxes_np is None or len(boxes_np) == 0:
                abs_boxes = np.empty((0, 4), dtype=float)
                abs_centroids = np.empty((0, 2), dtype=float)
            else:
                offset = np.array([x, y, x, y])
                abs_boxes = boxes_np + offset[None, :]
                abs_centroids = np.array([((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2) for bb in abs_boxes])

            result.abs_boxes = abs_boxes
            result.abs_centroids = abs_centroids
            all_results.append(result)

            # overlay masks; if none, just use the original padded patch
            masks_data = getattr(result.masks, "data", None)
            if masks_data is None:
                overlay = padded
            else:
                masks = masks_data.cpu().numpy() if hasattr(masks_data, "cpu") else masks_data
                cls_ids = (
                    result.boxes.cls.cpu().numpy().astype(int)
                    if hasattr(result.boxes.cls, "cpu")
                    else result.boxes.cls.astype(int)
                )

                overlay = padded.copy()
                for mask, cls in zip(masks, cls_ids):
                    mask = mask > 0.5
                    color = class_colors.get(int(cls), (255, 255, 255))
                    overlay_mask = overlay.copy()
                    overlay_mask[mask] = color
                    overlay = cv2.addWeighted(overlay, 1 - alpha, overlay_mask, alpha, 0)

            # write only the valid (unpadded) region back
            annotated[y:y_end, x:x_end] = overlay[:ph, :pw]


        # at this point, only up to `max_patches_in_memory` patches were in RAM/GPU for this loop
        # next batch will reuse memory

    # Output paths
    sample_name, _ = os.path.splitext(os.path.basename(image_path))
    sample_output_folder = os.path.join(output_path, sample_name)
    os.makedirs(sample_output_folder, exist_ok=True)

    result_image_path = os.path.join(sample_output_folder, f"{sample_name}.png")
    result_csv_path   = os.path.join(sample_output_folder, f"{sample_name}.csv")
    result_json_path  = os.path.join(sample_output_folder, f"{sample_name}.json")

    # Save outputs
    if not cv2.imwrite(result_image_path, annotated):
        raise IOError(f"Failed to save annotated image: {result_image_path}")

    if not save_segmentation_results(results=all_results, names=getattr(model, "names", {}), csv_path=result_csv_path, xml_path=xml_path):
        raise IOError(f"Failed to save CSV: {result_csv_path}")

    if not save_segmentation_results_json(results=all_results, names=getattr(model, "names", {}), json_path=result_json_path, xml_path=xml_path):
        raise IOError(f"Failed to save JSON: {result_json_path}")

    return {
        "image": result_image_path,
        "csv": result_csv_path,
        "json": result_json_path,
        "folder": sample_output_folder,
    }


# --- minimal main for quick testing ---
def _main_():
    paths = run_inference(
        image_path="data/20250813_A1-1_Glass/20250813_A1-1_U$3D.bmp",
        xml_path="data/20250813_A1-1_Glass/info.xml",
        model_path="models/IFM_segmentation.pt",
        output_path="output",
        patch_size=640,
        max_patches_in_memory=16,  # try different values based on RAM/GPU
    )
    print("Results:", paths)


if __name__ == "__main__":
    _main_()
