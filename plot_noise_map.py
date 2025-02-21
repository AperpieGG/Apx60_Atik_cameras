import argparse

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from astropy.io import fits
from utils_apx60 import plot_images


def read_bias_data(path, gain):
    """
    Reads bias images, filters them based on the gain setting, and computes per-pixel standard deviation.

    Parameters:
        path (str): Path to bias image directory.
        gain (float): Gain setting to filter images.

    Returns:
        tuple: (mean frame, std frame) where each pixel represents the mean and standard deviation.
    """
    list_images = sorted(glob.glob(os.path.join(path, 'bias*.fits')))
    if not list_images:
        print(f"[ERROR] No bias images found in {path}.")
        return None, None

    valid_images = []
    for image_path in list_images:
        header = fits.getheader(image_path)
        header_gain = header.get('CAM-GAIN', None)
        if header_gain is not None and np.isclose(header_gain, gain, atol=1e-3):
            valid_images.append(image_path)

    if not valid_images:
        print(f"[ERROR] No valid images found for gain {gain}.")
        return None, None

    bias_values = []
    for image_path in valid_images:
        with fits.open(image_path, memmap=True) as hdulist:
            image_data = hdulist[0].data.astype(float)
            height, width = image_data.shape
            trimmed_image = image_data[100:height - 100, 100:width - 100]
            bias_values.append(trimmed_image)

    bias_values = np.array(bias_values)
    value_mean = np.mean(bias_values, axis=0)  # Mean per pixel
    value_std = np.std(bias_values, axis=0)  # Std deviation (noise) per pixel

    return value_mean, value_std


def plot_noise_frame(value_std, gain, save_path):
    """
    Plots the reconstructed noise frame where each pixel represents its standard deviation.

    Parameters:
        value_std (numpy.ndarray): 2D array representing per-pixel noise (std deviation).
        gain (int): Gain value used for filtering.
        save_path (str): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Display the noise frame as a heatmap
    img = ax.imshow(value_std, cmap='inferno', origin='lower')

    # Add a colorbar indicating the standard deviation values
    cbar = plt.colorbar(img, ax=ax, label='Standard Deviation (ADU)')

    ax.set_title(f'Pixel Noise Map (Gain {gain})')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.tight_layout()
    # Save figure
    save_filename = os.path.join(save_path, f'read_noise_map_{gain}.png')
    plt.savefig(save_filename)
    print(f'[INFO] Read Noise plot saved: {save_filename}')

    plt.show()


def main():
    """
    Main function to parse command-line arguments, read bias data, and plot read noise.
    """
    plot_images()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process bias images to calculate read noise')
    parser.add_argument('directory', type=str, help='Directory containing bias images (relative to base path).')
    parser.add_argument('gain', type=int, help='Gain value of the camera.')
    parser.add_argument('--row_banding', default=False, action='store_true',
                        help='Apply a Row-to-Row banding correction on bias images before rn.')
    args = parser.parse_args()

    base_path = '/data/'
    save_path = '/home/ops/Downloads/RN_apx_R-R/'
    path = os.path.join(base_path, args.directory)

    if not os.path.exists(path):
        print(f'[ERROR] Directory {path} does not exist.')
        return

    # === Load Bias Data ===
    _, value_std = read_bias_data(base_path, args.gain_setting)

    # === Plot Reconstructed Noise Frame ===
    if value_std is not None:
        plot_noise_frame(value_std, args.gain_setting, save_path)
    else:
        print("[ERROR] Could not generate noise frame.")