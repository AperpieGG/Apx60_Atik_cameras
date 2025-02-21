#! /usr/bin/env python
"""
This script reads the bias data from the given directory.
It then plots the binary image of the noise pixels and the time series of the most noisy pixel.
The noise pixels will be the ones that are equal or above the threshold value.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import argparse


def plot_images():
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 12


def read_bias_data(path, gain):
    """
    Reads bias images from the directory, filters by the correct gain setting,
    applies optional row banding correction, and calculates mean and standard deviation.

    Parameters:
        path (str): The directory containing bias images.
        gain (float): The expected gain value from the camera.
    Returns:
        tuple: (mean values, standard deviation values), or (None, None) if no valid images found.
    """
    list_images = glob.glob(os.path.join(path, 'bias*.fits'))
    print(f'Found {len(list_images)} images in {path}.')

    if not list_images:
        print(f"[ERROR] No bias images found in {path}.")
        return None, None

    # **Pre-filter images by checking only their headers**
    valid_images = []
    for image_path in list_images:
        try:
            header = fits.getheader(image_path)
            header_gain = header.get('CAM-GAIN', None)

            if header_gain is None:
                print(f"[WARNING] Missing 'CAM-GAIN' in {image_path}. Skipping.")
                continue

            if not np.isclose(header_gain, gain, atol=1e-3):  # Floating-point tolerance
                print(f"[WARNING] GAIN mismatch in {image_path} (Header: {header_gain}, Expected: {gain}). Skipping.")
                continue

            valid_images.append(image_path)

        except Exception as e:
            print(f"[ERROR] Could not read header from {image_path}: {e}")
            continue

    print(f"Filtered to {len(valid_images)} images with matching gain.")

    if not valid_images:
        print("[ERROR] No valid images found after filtering. Exiting.")
        return None, None

    # **Now process only valid images**
    bias_values = []
    for image_path in valid_images:
        with fits.open(image_path, memmap=False) as hdulist:
            image_data = hdulist[0].data.astype(float)

            # Trim by 100 pixels on all sides
            height, width = image_data.shape
            trimmed_image = image_data[1:height - 1, 1:width - 1]

            print(
                f"[INFO] Processing: {image_path} | Header Gain: {header_gain} | Trimmed Shape: {trimmed_image.shape}")
            bias_values.append(trimmed_image)

    if not bias_values:
        print("[ERROR] No valid bias images after reading. Exiting.")
        return None, None

    bias_values = np.array(bias_values)
    value_std = np.std(bias_values, axis=0).flatten()

    return bias_values, value_std


def find_pixel_coordinates(bias_data, threshold):
    # use bias data to extract the shape
    stds_1 = np.std(bias_data, axis=0).reshape(bias_data.shape[1], bias_data.shape[2])
    find = np.where(stds_1 > threshold)
    pixel_coordinates = np.array(find).T
    print('Number of noise pixels:', len(pixel_coordinates))

    fig_3 = plt.figure(figsize=(8, 8))
    plt.scatter(pixel_coordinates[:, 1], pixel_coordinates[:, 0], s=1, c='red', label='Pixel coordinates')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    percent = len(pixel_coordinates) / (bias_data.shape[1] * bias_data.shape[2]) * 100
    print("The pixels with coordinates {} are noise pixels".format(pixel_coordinates))
    plt.ylim(0, bias_data.shape[1])
    plt.xlim(0, bias_data.shape[2])
    plt.title('Noise pixels: ' + str(round(percent, 2)) + '%')
    fig_3.tight_layout()
    plt.show()

    print('Noise pixels:', percent, '%')
    return pixel_coordinates


def create_binary_image(bias_data, threshold):
    stds = np.std(bias_data, axis=0).reshape(bias_data.shape[1], bias_data.shape[2])
    binary_image = np.zeros_like(stds)
    binary_image[stds > threshold] = 1
    coordinates = np.array(np.where(binary_image == 1)).T
    print("Number of noise pixels:", len(coordinates))
    for coord in coordinates:
        x, y = coord[0], coord[1]
        value = stds[x, y]
        # print("Pixel at coordinates ({}, {}) has value {:.2f}.".format(x, y, value))
    return binary_image


def plot_binary_image(binary_image, gain):
    fig, ax = plt.subplots()
    im = ax.imshow(binary_image, cmap='Reds', origin='lower', extent=[0, binary_image.shape[0], 0,
                                                                      binary_image.shape[1]], vmin=0, vmax=1)
    ax.set_title('Noise pixels: ' + str(round(np.sum(binary_image) / (binary_image.shape[0] * binary_image.shape[1])
                                              * 100, 5)) + '%', fontsize=12)
    fig.colorbar(im, ax=ax, label='Threshold')
    plt.tight_layout()

    plt.savefig(f'Binary_image_{gain}.pdf', bbox_inches='tight')


def extract_pixel_time_series(bias_data, coordinates):
    frame_numbers = np.arange(bias_data.shape[0])
    pixel_values = bias_data[:, coordinates[0], coordinates[1]]
    return frame_numbers, pixel_values


def plot_pixel_time_series(frame_numbers, pixel_values, coordinates, gain):
    plt.figure()

    plt.plot(frame_numbers, pixel_values, 'o',
             label="RMS = " + str(round(np.std(pixel_values), 2)) + ' ADU')
    plt.plot(frame_numbers, pixel_values, '-', alpha=0.2)
    plt.xlabel('Frame Number')
    plt.ylabel('Pixel Value (ADU)')
    plt.title(f'Gain {gain} - Time Series for Pixel ({coordinates[0]}, {coordinates[1]})')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'Time_series_{gain}.pdf', bbox_inches='tight')


def main():
    """Main function to parse arguments and execute PTC analysis."""
    parser = argparse.ArgumentParser(description='Test average of each image and plot vs frame number')
    parser.add_argument('directory', type=str, help='Directory containing bias images (relative to base path).')
    parser.add_argument('gain_setting', type=int, help='Gain setting of the camera.')
    parser.add_argument('threshold', type=int, help='value times the read noise, ie. 3 sigma with sigma rn.')

    args = parser.parse_args()

    base_path = '/data/'
    save_path = '/home/ops/Downloads/RTS/'
    path = os.path.join(base_path, args.directory)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_images()
    # 3-D array of the bias images
    bias_values, value_std = read_bias_data(path)

    # Set threshold (times) the read noise
    threshold = value_std * args.threshold

    binary_image = create_binary_image(bias_values, threshold)
    plot_binary_image(binary_image, args.gain_setting)

    # Find coordinates of pixel with maximum standard deviation
    stds = np.std(bias_values, axis=0).reshape(bias_values.shape[1], bias_values.shape[2])
    max_std_index = np.unravel_index(np.argmax(stds), stds.shape)
    print("Coordinates of pixel with maximum standard deviation:", max_std_index)

    # Extract time series for the pixel with maximum standard deviation
    frame_numbers, pixel_values = extract_pixel_time_series(bias_values, max_std_index)
    plot_pixel_time_series(frame_numbers, pixel_values, max_std_index, args.gain_setting)

    # Print the total number of pixels that exceed the threshold value
    num_noise_pixels = np.sum(stds > threshold)
    print(f'Total number of pixels exceeding the threshold value: {num_noise_pixels}')


if __name__ == "__main__":
    main()
