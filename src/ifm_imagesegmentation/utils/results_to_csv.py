import pandas as pd
from ifm_imagesegmentation.utils.pixelToRealWorld import (
    calculate_pixels_per_mm,
    pixel_to_square_mm,
)


def save_segmentation_results(
    counts_from_mask, results, names, csv_path, xml_path=None, pixel_size=None
):
    """
    Save segmentation results from an Ultralytics model to a CSV summary.
    Even if no instances are found, a CSV with all classes and 0 values will be created.
    Additionally converts pixel areas to square millimeters using IFM XML calibration.
    Either xml_path or pixel_size should be provided. If both are provided,
    xml_path is used for calculations

    Args:
        counts_from_mask(list): list of total number of pixel for each defect type
        results (list): List of Ultralytics Results objects.
        names (dict): Mapping from class IDs to class names (model.names).
        csv_path (str): Path where the CSV summary will be saved.
        xml_path (str) (optional): Path to the Alicona/IFM XML file for pixel-to-mm calibration.
        pixel_size (float) (optional): Pixel size in meters
    Returns:
        bool: True if CSV was created successfully, False otherwise.
    """
    # --- Load calibration ---
    # Pixels per millimeter
    if xml_path is None and pixel_size is None:
        print("Error: no information on pixel size provided")
        return False
    elif xml_path is not None:
        px_per_mm = calculate_pixels_per_mm(xml_path)
    else:
        px_per_mm = 0.001 / pixel_size  # type: ignore
    # Pixels per square millimeter
    px_per_sq_mm = px_per_mm**2

    rows = []

    # 1) Collect raw data (pixels and convert to mm² immediately)
    for result in results:
        masks_data = getattr(result.masks, "data", None)
        if masks_data is None:
            continue

        classes = (
            result.boxes.cls.cpu().numpy().astype(int)
            if hasattr(result.boxes.cls, "cpu")
            else result.boxes.cls.astype(int)
        )

        for cls in classes:
            rows.append(
                {
                    "class_id": cls,
                    "class_name": names.get(cls, str(cls)),
                }
            )

    # 2) Create DataFrame for all classes (even those not detected)
    classes_df = pd.DataFrame(
        [
            {"class_id": cls_id, "class_name": cls_name}
            for cls_id, cls_name in names.items()
        ]
    )

    # 3) Summarize or fill default values
    if rows:
        df = pd.DataFrame(rows)
        summary = (
            df.groupby(["class_id", "class_name"])
            .agg(
                instance_count=("class_id", "size"),
            )
            .reset_index()
        )
        # Merge missing classes and fill NaN with 0
        summary = classes_df.merge(
            summary, on=["class_id", "class_name"], how="left"
        ).fillna({"instance_count": 0})
        # Ensure counts are integers
        summary["instance_count"] = summary["instance_count"].astype(int)
    else:
        # No instances found → set all classes with 0 values
        summary = classes_df.copy()
        summary["instance_count"] = 0

    summary["total_pixels"] = counts_from_mask[: len(names.items())]
    summary["total_area_mm2"] = pixel_to_square_mm(
        counts_from_mask[: len(names.items())], px_per_sq_mm
    )

    # 4) safe CSV
    try:
        summary.to_csv(csv_path, index=False)
        print(f"Segmentation summary saved to: {csv_path}")
        return True
    except Exception as e:
        print(f"Error saving CSV to '{csv_path}': {e}")
        return False
