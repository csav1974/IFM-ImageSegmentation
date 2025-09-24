import pandas as pd
from utils.pixelToRealWorld import calculate_pixels_per_mm, pixel_to_square_mm

def save_segmentation_results(results, names, csv_path, xml_path):
    """
    Save segmentation results from an Ultralytics model to a CSV summary.
    Even if no instances are found, a CSV with all classes and 0 values will be created.
    Additionally converts pixel areas to square millimeters using IFM XML calibration.

    Args:
        results (list): List of Ultralytics Results objects.
        names (dict): Mapping from class IDs to class names (model.names).
        csv_path (str): Path where the CSV summary will be saved.
        xml_path (str): Path to the Alicona/IFM XML file for pixel-to-mm calibration.
    Returns:
        bool: True if CSV was created successfully, False otherwise.
    """
    # --- Load calibration ---
    # Pixels per millimeter
    px_per_mm = calculate_pixels_per_mm(xml_path)
    # Pixels per square millimeter
    px_per_sq_mm = px_per_mm ** 2

    rows = []

    # 1) Collect raw data (pixels and convert to mm² immediately)
    for result in results:
        masks_data = getattr(result.masks, 'data', None)
        if masks_data is None:
            continue

        masks = (masks_data.cpu().numpy()
                    if hasattr(masks_data, 'cpu')
                    else masks_data)
        classes = (result.boxes.cls.cpu().numpy().astype(int)
                   if hasattr(result.boxes.cls, 'cpu')
                   else result.boxes.cls.astype(int))

        for cls, mask in zip(classes, masks):
            pixel_area = int(mask.sum())
            area_mm2 = pixel_to_square_mm(pixel_area, px_per_sq_mm)
            rows.append({
                'class_id': cls,
                'class_name': names.get(cls, str(cls)),
                'pixel_area': pixel_area,
                'area_mm2': area_mm2
            })

    # 2) Create DataFrame for all classes (even those not detected)
    classes_df = pd.DataFrame([
        {'class_id': cls_id, 'class_name': cls_name}
        for cls_id, cls_name in names.items()
    ])

    # 3) Summarize or fill default values
    if rows:
        df = pd.DataFrame(rows)
        summary = (
            df.groupby(['class_id', 'class_name'])
              .agg(
                  instance_count=('pixel_area', 'size'),
                  total_pixels=('pixel_area', 'sum'),
                  total_area_mm2=('area_mm2', 'sum')
              )
              .reset_index()
        )
        # Merge missing classes and fill NaN with 0
        summary = (
            classes_df
            .merge(summary, on=['class_id', 'class_name'], how='left')
            .fillna({'instance_count': 0, 'total_pixels': 0, 'total_area_mm2': 0})
        )
        # Ensure counts are integers
        summary['instance_count'] = summary['instance_count'].astype(int)
        summary['total_pixels'] = summary['total_pixels'].astype(int)
    else:
        # No instances found → set all classes with 0 values
        summary = classes_df.copy()
        summary['instance_count'] = 0
        summary['total_pixels'] = 0
        summary['total_area_mm2'] = 0.0

    # 4) safe CSV
    try:
        summary.to_csv(csv_path, index=False)
        print(f"Segmentation summary saved to: {csv_path}")
        return True
    except Exception as e:
        print(f"Error saving CSV to '{csv_path}': {e}")
        return False
