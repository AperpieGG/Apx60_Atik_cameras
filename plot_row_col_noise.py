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


def read_images(path):
    """
    Reads bias images from the directory and calculates mean and standard deviation.

    Parameters:
        path (str): The directory containing bias images.

    Returns:
        tuple: (mean values, standard deviation values), or (None, None) if no valid images found.
    """
    list_images = glob.glob(os.path.join(path, 'image*.fits'))
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

    return bias_values


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


def plot_col_noise(std_col, mean_col, save_path):
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0)

    ax1 = plt.subplot(gs[0])
    hb = ax1.hist2d(mean_col, std_col, bins=50, cmap='cividis', norm=LogNorm())
    ax1.set_xlabel('Mean (ADU)')
    ax1.set_ylabel('Column for Row (ADU)')
    ax1.set_ylim(0, 0.5)

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
    save_filename = os.path.join(save_path, f'column_noise.pdf')
    plt.savefig(save_filename, bbox_inches='tight')
    print(f'[INFO] Read Noise plot saved: {save_filename}')

    plt.show()


def plot_row_noise(std_row, mean_row, save_path):
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0)

    ax1 = plt.subplot(gs[0])
    hb = ax1.hist2d(mean_row, std_row, bins=50, cmap='cividis', norm=LogNorm())
    ax1.set_xlabel('Mean (ADU)')
    ax1.set_ylabel('Noise for Row (ADU)')
    ax1.set_ylim(0, 1)

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
    save_filename = os.path.join(save_path, f'row_noise.pdf')
    plt.savefig(save_filename, bbox_inches='tight')
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
    args = parser.parse_args()

    base_path = '/Users/u5500483/Downloads/'
    save_path = '/Users/u5500483/Downloads/'
    path = os.path.join(base_path, args.directory)

    if not os.path.exists(path):
        print(f'[ERROR] Directory {path} does not exist.')
        return

    bias_values = read_images(path)
    std_row, mean_row = row_to_row(bias_values)
    std_col, mean_col = column_to_column(bias_values)
    plot_row_noise(std_row, mean_row, save_path)
    plot_col_noise(std_col, mean_col, save_path)


if __name__ == '__main__':
    main()
