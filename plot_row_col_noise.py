import argparse
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from utils_apx60 import plot_images

plot_images()


def read_images(path, gain):
    """
    Reads bias images from the directory, filters by the correct gain setting,
    and calculates mean and standard deviation.

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
        with fits.open(image_path, memmap=True) as hdulist:
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

    return bias_values


def read_bias_data(path):
    """
    Reads bias images from the directory, filters by the correct gain setting,
    applies optional row banding correction, and calculates mean and standard deviation.

    Parameters:
        path (str): The directory containing bias images.

    Returns:
        tuple: (mean values, standard deviation values), or (None, None) if no valid images found.
    """
    list_images = glob.glob(os.path.join(path, 'simage*.fits'))
    print(f'Found {len(list_images)} images in {path}.')

    if not list_images:
        print(f"[ERROR] No bias images found in {path}.")
        return None, None

    # **Now process only valid images**
    bias_values = []
    for image_path in list_images:
        with fits.open(image_path, memmap=False) as hdulist:
            image_data = hdulist[0].data.astype(float)
            bias_values.append(image_data)

    if not bias_values:
        print("[ERROR] No valid bias images after reading. Exiting.")
        return None, None

    bias_values = np.array(bias_values)
    std_row, mean_row = row_to_row(bias_values)
    std_col, mean_col = column_to_column(bias_values)

    return std_row, std_col, mean_row, mean_col


def row_to_row(bias_values):
    row_means = np.mean(bias_values[:, :, :], axis=2)  # Mean along columns for each row
    std_val = np.std(row_means, axis=0).flatten()  # 0 for 2048x100, 1 for 100
    mean_val = np.mean(row_means, axis=0).flatten()
    return std_val, mean_val


def column_to_column(bias_values):
    col_means = np.mean(bias_values[:, :, :], axis=1)  # Mean along rows for each column
    std_val = np.std(col_means, axis=0).flatten()  # 0 for 2048x100, 1 for 100
    mean_val = np.mean(col_means, axis=0).flatten()
    return std_val, mean_val


def plot_col_noise(std_col, mean_col, gain, save_path):
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0)

    ax1 = plt.subplot(gs[0])
    hb = ax1.hist2d(mean_col, std_col, bins=50, cmap='cividis', norm=LogNorm())
    ax1.set_xlabel('Mean (ADU)')
    ax1.set_ylabel('Noise for Row (ADU)')

    ax2 = plt.subplot(gs[1])
    ax2.hist(std_col, bins=50, orientation='horizontal', color='blue', histtype='step')
    ax2.set_xlabel('Number of Pixels')
    ax2.set_xscale('log')

    ax2.axhline(np.mean(std_col), color='green', linestyle=':')

    ax2.yaxis.set_ticklabels([])

    y_min, y_max = ax1.get_ylim()
    ax2.set_ylim(y_min, y_max)

    fig.colorbar(ax1.get_children()[0], ax=ax2, label='Number of Pixels')
    fig.tight_layout()
    # Save figure
    save_filename = os.path.join(save_path, f'column_noise_{gain}.png')
    plt.savefig(save_filename)
    print(f'[INFO] Read Noise plot saved: {save_filename}')

    plt.show()


def plot_row_noise(std_row, mean_row, gain, save_path):
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0)

    ax1 = plt.subplot(gs[0])
    hb = ax1.hist2d(mean_row, std_row, bins=50, cmap='cividis', norm=LogNorm())
    ax1.set_xlabel('Mean (ADU)')
    ax1.set_ylabel('Noise for Row (ADU)')

    ax2 = plt.subplot(gs[1])
    ax2.hist(std_row, bins=50, orientation='horizontal', color='blue', histtype='step')
    ax2.set_xlabel('Number of Pixels')
    ax2.set_xscale('log')

    ax2.axhline(np.mean(std_row), color='green', linestyle=':')

    ax2.yaxis.set_ticklabels([])

    y_min, y_max = ax1.get_ylim()
    ax2.set_ylim(y_min, y_max)

    fig.colorbar(ax1.get_children()[0], ax=ax2, label='Number of Pixels')
    fig.tight_layout()
    # Save figure
    save_filename = os.path.join(save_path, f'row_noise_{gain}.png')
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
    args = parser.parse_args()

    base_path = '/data/20250201/'
    save_path = '/home/ops/Downloads/RN_apx60/'
    path = os.path.join(base_path, args.directory)

    if not os.path.exists(path):
        print(f'[ERROR] Directory {path} does not exist.')
        return

    bias_values = read_images(path, args.gain)
    std_row, mean_row = row_to_row(bias_values)
    std_col, mean_col = column_to_column(bias_values)
    plot_row_noise(std_row, mean_row, args.gain, save_path)
    plot_col_noise(std_col, mean_col, args.gain, save_path)


if __name__ == '__main__':
    main()