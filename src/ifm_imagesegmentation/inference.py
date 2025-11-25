"""
This script performs image segmentation inference using a YOLO model on a list of input images, overlays segmentation masks, and saves results in annotated images, CSV, and JSON formats.

Main Features:
- Loads a YOLO segmentation model from a specified path.
- Processes each image in fixed-size patches to handle large images.
- Applies model predictions to each patch, overlays segmentation masks with configurable colors and opacity.
- Adjusts bounding boxes and centroids to absolute image coordinates.
- Saves annotated images, CSV, and JSON results for each input image.
- Ensures input image and XML measurement file lists are aligned.

Configuration:
- MODEL_PATH: Path to the trained YOLO segmentation model.
- IMAGE_PATHS: List of image file paths to process.
- XML_PATHS: List of corresponding XML measurement file paths.
- OUTPUT_PATH: Directory to save output results.
- PATCH_SIZE: Size of image patches for model input.
- class_colors: Dictionary mapping class indices to BGR colors for mask overlays.
- alpha: Opacity of mask overlays.

Dependencies:
- ultralytics (YOLO)
- OpenCV (cv2)
- NumPy
- Custom utility modules for saving results and post-processing.

Usage:
- Configure paths and parameters as needed.
- Run the script to process all images and save results in the specified output directory.
"""


from ultralytics import YOLO
import cv2
import numpy as np
import os
from ifm_imagesegmentation.utils.results_to_csv import save_segmentation_results
from ifm_imagesegmentation.utils.results_to_JSON import save_segmentation_results_json

# ----------------------------
# Configuration (change as needed)
# ----------------------------
MODEL_PATH = "models/IFM_segmentation.pt"
# Lists of input images and corresponding XML measurement files
IMAGE_PATHS = [
    "data/IFM_texture_cut.bmp"
    # Add more image paths here, e.g. "data/another/sample.bmp",
]
XML_PATHS = [
    "data/IFM_texture_cut_info.xml",
    # Add more XML paths here, ensuring the order matches IMAGE_PATHS
]
# Path to output folder
OUTPUT_PATH = "output"
# Expected image size for the model
PATCH_SIZE = 640
# Define a BGR color for each class index the model outputs.
class_colors = {
    0: (0,   0,   255),   # class 0 → red   (here Chipping)
    1: (0,   255, 0  ),   # class 1 → green (here Scratching)
    2: (255, 0,   0  ),   # class 2 → blue  (here Whiskers)
}
# Mask overlay opacity
alpha = 0.7  # 0.0 = fully transparent, 1.0 = fully opaque
max_size = 512  # size for printed mask


def run_image_segmentation(image_path, model_path, output_path, patch_size, xml_path=None, pixel_size=None):
    model = YOLO(model_path)
    # Load the original image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image from {image_path}")

    height, width = original.shape[:2]
    annotated = original.copy()
    all_results = []

    # Process in fixed-size patches
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            patch = original[y:y_end, x:x_end]

            # Pad if needed
            ph, pw = patch.shape[:2]
            if ph < patch_size or pw < patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                padded[:ph, :pw] = patch
            else:
                padded = patch

            # Model prediction
            results = model.predict(
                padded,
                imgsz=patch_size,
                augment=False,
                conf=0.4,
                iou=0.3,
                )
            
            result = results[0]

            # Absolute bounding boxes and centroids
            boxes = result.boxes.xyxy
            boxes_np = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else boxes
            offset = np.array([x, y, x, y])
            abs_boxes = boxes_np + offset[None, :]

            abs_centroids = np.array([((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2) for bb in abs_boxes])
            result.abs_boxes = abs_boxes
            result.abs_centroids = abs_centroids

            all_results.append(result)

            # Overlay masks if present
            masks_data = getattr(result.masks, 'data', None)
            if masks_data is None:
                continue
            masks = masks_data.cpu().numpy() if hasattr(masks_data, 'cpu') else masks_data
            cls_ids = result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes.cls, 'cpu') else result.boxes.cls.astype(int)

            overlay = padded.copy()
            for mask, cls in zip(masks, cls_ids):
                mask = mask > 0.5
                color = class_colors.get(cls, (255, 255, 255))
                overlay_mask = overlay.copy()
                overlay_mask[mask] = color
                overlay = cv2.addWeighted(overlay, 1 - alpha, overlay_mask, alpha, 0)

            # Place annotated patch back
            annotated[y:y_end, x:x_end] = overlay[:ph, :pw]
    
    # Prepare output paths
    sample_name, _ = os.path.splitext(os.path.basename(image_path))
    # sample_output_folder = os.path.join(output_path, sample_name)
    os.makedirs(output_path, exist_ok=True)

    result_image_path = os.path.join(output_path, f"{sample_name}_prediction.png")
    result_csv_path = os.path.join(output_path, f"{sample_name}_prediction.csv")
    result_json_path = os.path.join(output_path, f"{sample_name}_prediction.json")

    # Save annotated image
    if not cv2.imwrite(result_image_path, annotated):
        raise IOError(f"Could not save segmented image to {result_image_path}")

    # Save CSV and JSON results
    if not save_segmentation_results(results=all_results, names=model.names, csv_path=result_csv_path, xml_path=xml_path, pixel_size=pixel_size):
        raise IOError(f"Could not save CSV results to {result_csv_path}")
    if not save_segmentation_results_json(results=all_results, names=model.names, json_path=result_json_path, xml_path=xml_path, pixel_size=pixel_size):
        raise IOError(f"Could not save JSON results to {result_json_path}")

    print(f"Processed '{image_path}', results saved in '{output_path}'")


if __name__ == "__main__":
    # Ensure image and XML lists align
    if len(IMAGE_PATHS) != len(XML_PATHS):
        raise ValueError("IMAGE_PATHS and XML_PATHS must have the same number of entries")
    
    for image_path, xml_path in zip(IMAGE_PATHS, XML_PATHS):
        run_image_segmentation(image_path=image_path, model_path=MODEL_PATH, output_path=OUTPUT_PATH, patch_size=PATCH_SIZE, xml_path=xml_path)
