import json
import pandas as pd
from utils.pixelToRealWorld import calculate_pixels_per_mm, pixel_to_square_mm

def save_segmentation_results_json(results, names, json_path, xml_path):
    """
    Save detailed segmentation results from an Ultralytics model to a JSON file,
    including pixel areas, converted mm² values, and absolute positions (bbox + centroid).

    Args:
        results (list): List of Ultralytics Results objects, each with `abs_boxes` and `abs_centroids` attributes.
        names (dict): Mapping from class IDs to class names (model.names).
        json_path (str): Path where the JSON file will be saved.
        xml_path (str): Path to the Alicona/IFM XML file for pixel-to-mm calibration.
    Returns:
        bool: True if JSON was created successfully, False otherwise.
    """
    # 1) Load calibration
    px_per_mm = calculate_pixels_per_mm(xml_path)
    px_per_sq_mm = px_per_mm ** 2

    objects = []

    # 2) Collect data per detected instance
    for result in results:
        masks_data = getattr(result.masks, 'data', None)
        if masks_data is None:
            continue

        # convert masks and classes to numpy
        masks = (masks_data.cpu().numpy() 
                 if hasattr(masks_data, 'cpu') 
                 else masks_data)
        classes = (result.boxes.cls.cpu().numpy().astype(int)
                   if hasattr(result.boxes.cls, 'cpu')
                   else result.boxes.cls.astype(int))

        # absolute positions
        abs_boxes = getattr(result, 'abs_boxes', None)
        abs_centroids = getattr(result, 'abs_centroids', None)

        # iterate instances
        for idx, (cls, mask) in enumerate(zip(classes, masks)):
            cls = int(cls)
            pixel_area = int(mask.sum())
            area_mm2 = float(pixel_to_square_mm(pixel_area, px_per_sq_mm))

            obj = {
                'class_id': cls,
                'class_name': names.get(cls, str(cls)),
                'pixel_area': pixel_area,
                'area_mm2': area_mm2
            }

            # include absolute bbox if available
            if abs_boxes is not None:
                bb = abs_boxes[idx]
                obj['bbox'] = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
            # include centroid if available
            if abs_centroids is not None:
                ct = abs_centroids[idx]
                obj['centroid'] = [float(ct[0]), float(ct[1])]
                obj['centroid_mm'] = [float(ct[0]) / px_per_mm, float(ct[1]) / px_per_mm]
            objects.append(obj)

    # 3) Write to JSON
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(objects, f, indent=2, ensure_ascii=False)
        print(f"Detailed segmentation results saved to: {json_path}")
        return True
    except Exception as e:
        print(f"Error saving JSON to '{json_path}': {e}")
        return False
