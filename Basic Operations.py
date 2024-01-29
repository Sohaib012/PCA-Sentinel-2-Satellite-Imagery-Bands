import rasterio
from rasterio.plot import show
import numpy as np

bands = ["T43SBT_20231116T055109_AOT_20m.jp2",
"T43SBT_20231116T055109_B01_20m.jp2",
"T43SBT_20231116T055109_B02_20m.jp2",
"T43SBT_20231116T055109_B03_20m.jp2",
"T43SBT_20231116T055109_B04_20m.jp2",
"T43SBT_20231116T055109_B05_20m.jp2",
"T43SBT_20231116T055109_B06_20m.jp2",
"T43SBT_20231116T055109_B07_20m.jp2",
"T43SBT_20231116T055109_B8A_20m.jp2",
"T43SBT_20231116T055109_B11_20m.jp2",
"T43SBT_20231116T055109_B12_20m.jp2",
"T43SBT_20231116T055109_SCL_20m.jp2",
"T43SBT_20231116T055109_TCI_20m.jp2",
"T43SBT_20231116T055109_WVP_20m.jp2"
]
# Define the output filename prefix
output_filename_prefix = "D:/GIKI/5th semester/ES-304 -Linear Algebra 2/CEP/out/ModifiedImage_band"

# List to store individual bands
bands_list = []

# Iterate through bands
for idx, band in enumerate(bands):
    filepath = f"D:/GIKI/5th semester/ES-304 -Linear Algebra 2/CEP/S2B_MSIL2A_20231116T055109_N0509_R048_T43SBT_20231116T082205.SAFE/GRANULE/L2A_T43SBT_A034966_20231116T055536/IMG_DATA/R20m/{band}"

    with rasterio.open(filepath) as src:
        image = src.read()
        metadata = src.meta

    # Get band name
    band_name = f"Band {idx + 1}"

    # Visualize the band
    show(image[0], title=band_name)

    # Define the crop window (x_min, y_min, x_max, y_max)
    crop_window = (500, 1500, 4000, 5000)

    # Crop the image
    cropped_image = image[:, crop_window[0]:crop_window[2], crop_window[1]:crop_window[3]]

    concatenated_image = np.concatenate(cropped_image, axis=-1)
    # Append cropped image to the list
    bands_list.append(cropped_image)

# Save individual bands
for idx, band_data in enumerate(bands_list):
    output_filename = f"{output_filename_prefix}_band{idx + 1}.jp2"

    # Copy metadata and update as needed
    output_metadata = metadata.copy()
    output_metadata.update({
        "driver": "JP2OpenJPEG",
        "height": band_data.shape[1],
        "width": band_data.shape[2],
        "count": band_data.shape[0]
    })

    # Save the image
    with rasterio.open(output_filename, "w", **output_metadata) as dst:
        dst.write(band_data)