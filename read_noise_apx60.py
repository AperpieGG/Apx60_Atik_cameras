#!/usr/bin/env python

"""
This script reads the bias images from the given directory. It then plots
the read noise 2-D histogram and the histogram of the read noise values.
It will print the median, mean, and RMS of the read noise values.
"""

import argparse
import glob
import json
import os
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from utils import plot_images


def read_bias_data(path, gain):
    """
    Reads the bias images from the specified directory, verifies the gain setting in the headers,
    trims images, calculates the mean and standard deviation of bias values.

    Parameters:
        path (str): The path to the directory containing bias images.
        gain (int or float): The expected gain setting of the camera.

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
            trimmed_image = image_data[1000:height - 1000, 1000:width - 1000]

            print(
                f'[INFO] Processing: {image_path} | Header Gain: {header_gain} | Trimmed Shape: {trimmed_image.shape}')
            bias_values.append(trimmed_image)

    if not bias_values:
        print("[ERROR] No valid bias images after filtering. Exiting.")
        return None, None

    bias_values = np.array(bias_values)

    value_mean = np.mean(bias_values, axis=0).flatten()
    value_std = np.std(bias_values, axis=0).flatten()
    return value_mean, value_std


def plot_read_noise(value_mean, value_std, gain, save_path, sensitivity):
    """
    Plots the 2-D histogram of the mean and standard deviation of the read noise values.

    Parameters:
        value_mean (numpy.ndarray): The mean values of the read noise.
        value_std (numpy.ndarray): The standard deviation values of the read noise.
        gain (int or float): The gain setting of the camera.
        save_path (str): The path to save the plot.
        sensitivity (float): The sensitivity factor to convert ADU to electrons.
    """

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0)

    ax1 = plt.subplot(gs[0])
    hb = ax1.hist2d(value_mean, value_std, bins=1000, cmap='cividis', norm=LogNorm())
    ax1.set_xlabel('Mean (ADU)')
    ax1.set_ylabel('RMS (ADU)')
    ax1.set_title(f'Read Noise - Gain {gain}')

    # Add text to the plot
    value_median_hist = np.median(value_std)
    ax1.text(
        0.95, 0.95,
        f'Median RMS: {value_median_hist * sensitivity:.2f} ADU',
        transform=ax1.transAxes,
        ha='right', va='top',
        color='white', fontsize=12,
        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
    )

    ax2 = plt.subplot(gs[1])
    ax2.hist(value_std, bins=1000, orientation='horizontal', color='blue', histtype='step')
    ax2.set_xlabel('Number of Pixels')
    ax2.set_xscale('log')

    value_median = np.median(value_std)
    value_mean_std = np.mean(value_std)
    rms = np.sqrt(np.mean(value_std ** 2))

    print(f'Value Median = {value_median}')
    print(f'Value Mean = {value_mean_std}')
    print(f'RMS = {rms}')

    ax2.axhline(value_median, color='green', linestyle=':')
    ax2.yaxis.set_ticklabels([])

    y_min, y_max = ax1.get_ylim()
    ax2.set_ylim(y_min, y_max)

    fig.colorbar(ax1.get_children()[0], ax=ax2, label='Number of Pixels')
    fig.tight_layout()

    # Save figure
    save_filename = os.path.join(save_path, f'read_noise_{gain}.png')
    plt.savefig(save_filename)
    print(f'[INFO] Read Noise plot saved: {save_filename}')

    # Save results in a JSON file
    results = {
        'median': value_median,
        'mean': value_mean_std,
        'rms': rms
    }
    json_filename = os.path.join(save_path, f'read_noise_{gain}.json')
    with open(json_filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    print(f'[INFO] Results saved in: {json_filename}')

    plt.show()


def main():
    """
    Main function to parse command-line arguments, read bias data, and plot read noise.
    """
    plot_images()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process bias images to calculate read noise')
    parser.add_argument('directory', type=str, help='Directory containing bias images (relative to base path).')
    parser.add_argument('gain', type=float, help='Gain value of the camera.')
    args = parser.parse_args()

    base_path = '/data/'
    save_path = '/home/ops/Downloads/RN/'
    sensitivity = 1.0
    path = os.path.join(base_path, args.directory)

    if not os.path.exists(path):
        print(f'[ERROR] Directory {path} does not exist.')
        return

    # Read bias data
    value_mean, value_std = read_bias_data(path, args.gain)
    if value_mean is None or value_std is None:
        print("[ERROR] No valid data to process.")
        return

    # Generate and save read noise plot
    plot_read_noise(value_mean, value_std, args.gain, save_path, sensitivity)


if __name__ == "__main__":
    main()