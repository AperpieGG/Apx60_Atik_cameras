"""
This script takes bias images and does some statistics on them.
Measuring the row to row or column to column variation and prints the results
"""
import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from utils_apx60 import plot_images
import astropy.io.fits as fits
import glob


def find_sensitivity(path, gain):
    file = os.path.join(path, f'ptc_results_{gain}.json')
    print(f'[INFO] Reading Sensitivity from: {file}')
    try:
        with open(file, 'r') as f:
            data = json.load(f)

        # Extract required values
        sensitivity = data.get('gain', None)

        if sensitivity is None:
            print(f"[WARNING] 'gain' key missing in {file}. Exiting.")
            return None
        else:
            print(f'[INFO] Found Sensitivity for Gain_setting {gain}: {sensitivity} e-/ADU')
    except Exception as e:
        print(f"[ERROR] Could not process {file}. Error: {e}")
        return None

    return sensitivity


def get_images(path, gain, sensitivity):
    """
    Reads the bias images from the specified directory, verifies the gain setting in the headers,
    trims images, calculates the mean and standard deviation of bias values.

    Parameters:
        path (str): The path to the directory containing bias images.
        gain (int or float): The expected gain setting of the camera.
        sensitivity (int or float): The sensitivity (e-/ADU) of the camera.

    Returns:
        tuple: A tuple containing the mean values and standard deviation values.
    """
    list_images = glob.glob(os.path.join(path, 'bias*.fits'))
    print(f'Found {len(list_images)} images in {path}.')

    if not list_images:
        print(f"No bias images found in {path}")
        return None, None

    bias_values = []

    for image_path in list_images:
        with fits.open(image_path) as hdulist:
            image_data = hdulist[0].data.astype(float)
            header_gain = hdulist[0].header.get('CAM-GAIN', None)  # Extract gain from header

            # Check if the gain exists in the header
            if header_gain is None:
                print(f"[WARNING] Missing 'CAM-GAIN' in header of {image_path}. Skipping.")
                continue

            # Ensure the gain matches the expected value (with a small numerical tolerance)
            if not np.isclose(header_gain, gain, atol=1e-3):  # Allows for floating-point precision issues
                print(f"[WARNING] GAIN mismatch in {image_path} (Header: {header_gain}, Expected: {gain}). Skipping.")
                continue

            # Get image dimensions and trim by 100 pixels on all sides
            height, width = image_data.shape
            trimmed_image = image_data[100:height - 100, 100:width - 100]

            print(
                f'[INFO] Processing: {image_path} | Header Gain: {header_gain} | Trimmed Shape: {trimmed_image.shape}')
            bias_values.append(trimmed_image)

    if not bias_values:
        print("[ERROR] No valid bias images after filtering. Exiting.")
        return None, None

    return np.array(bias_values * sensitivity)


def row_to_row(bias_values):
    row_means = np.mean(bias_values[:, :, :], axis=2)  # Mean along columns for each row
    std_val = np.std(row_means, axis=0)  # 0 for 2048x100, 1 for 100
    return std_val


def column_to_column(bias_values):
    col_means = np.mean(bias_values[:, :, :], axis=1)  # Mean along rows for each column
    std_val = np.std(col_means, axis=0)  # 0 for 2048x100, 1 for 100
    return std_val


def plot_histograms(row_std, col_std, save_path, gain):

    plt.subplot(2, 1, 1)
    plt.hist(row_std, bins=15, alpha=0.8, label='Row-Row', density=True)
    plt.xlabel('Standard Deviation (e$^{-}$)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.hist(col_std, bins=15, alpha=0.8, label='Col-Col', density=True)
    plt.xlabel('Standard Deviation (e$^{-}$)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'R-R_{gain}.png'))
    print(f'[INFO] R-R plot saved: {os.path.join(save_path, f"R-R_{gain}.png")}')
    plt.tight_layout()


def save_results(path, gain, row_std, col_std):
    # Save results in a JSON file
    results = {
        'row_std_mean': np.mean(row_std),
        'col_std_mean': np.mean(col_std),
        'row_std_values': row_std.tolist(),
        'col_std_values': col_std.tolist()
    }
    json_filename = os.path.join(path, f'R-R_{gain}.json')
    with open(json_filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    print(f'[INFO] Results saved in: {json_filename}')


def main():
    """Main function to parse arguments and execute PTC analysis."""
    parser = argparse.ArgumentParser(description='Row to Row and Column to Column variation analysis on bias images.')
    parser.add_argument('gain_setting', type=int, help='Gain setting of the camera.')
    parser.add_argument('directory', type=str, help='Directory containing bias images'
                                                    ' (relative to base path, e.g. 20250201).')
    args = parser.parse_args()

    plot_images()

    base_path = '/data/'
    save_path = '/home/ops/Downloads/R-R_apx60/'
    path_ptc_results = '/home/ops/Downloads/PTC_apx60/'
    path = os.path.join(base_path, args.directory)

    sensitivity = find_sensitivity(path_ptc_results, args.gain_setting)

    bias_values = get_images(path, args.gain_setting, sensitivity)

    row_std = row_to_row(bias_values)
    col_std = column_to_column(bias_values)

    print(f'[INFO] R-R Row Mean: {np.mean(row_std)}')
    print(f'[INFO] R-R Column Mean: {np.mean(col_std)}')

    plot_histograms(row_std, col_std, save_path, args.gain_setting)

    save_results(save_path, args.gain_setting, row_std, col_std)


if __name__ == '__main__':
    main()
