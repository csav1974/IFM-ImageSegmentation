"""
This script is specific for the IFM used at the University of Innsbruck.
Change the xml handling to your needs or set a fixed parameter for the conversion in the functions at the top.
"""

import xml.etree.ElementTree as ET


def pixel_to_square_mm(pixel_count: int, pixel_per_square_mm: float = 316068.84):
    return pixel_count / pixel_per_square_mm


def pixel_to_mm(pixel_count: int, pixel_per_mm: float = 562.2):
    return pixel_count / pixel_per_mm


def calculate_pixels_per_mm(xml_file: str) -> float:
    """
    Reads the Alicona/IFM XML file, retrieves the pixel sizes (in meters/pixel)
    from the <pixelsize> tag, calculates the average of the x- and y-values,
    converts from meter to millimeter, and finally returns how many pixels
    fit into one millimeter (pixel/mm).

    :param xml_file: Path to the Alicona/IFM XML file.
    :return: Conversion factor in pixel/mm.
    """
    # Parse the XML file and get the root element
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the <vector> element within <pixelsize>
    pixel_vector = root.find(".//generalCalibrationData/pixelsize/vector")
    if pixel_vector is None:
        raise ValueError(
            "Could not find the <pixelsize>/<vector> element in the XML file."
        )

    # Split the text to get individual values
    px_str = pixel_vector.text.strip().split()
    if len(px_str) < 2:
        raise ValueError("Unexpected format: not enough pixel size values found.")

    # Convert the string values to float
    # (these values are in meters/pixel)
    px_x_m = float(px_str[0])  # pixel size in x-direction (m/pixel)
    px_y_m = float(px_str[1])  # pixel size in y-direction (m/pixel)

    # Compute the average pixel size in meters/pixel
    avg_px_m = (px_x_m + px_y_m) / 2.0

    # Convert from meter/pixel to millimeter/pixel
    avg_px_mm = avg_px_m * 1000.0

    # Convert mm/pixel to pixel/mm (inverse value)
    px_per_mm = 1.0 / avg_px_mm

    return px_per_mm


if __name__ == "__main__":
    # Example usage
    xml_file_path = "info.xml"
    px_mm_factor = calculate_pixels_per_mm(xml_file_path)
    print(f"{px_mm_factor:.6f} pixel/mm")
