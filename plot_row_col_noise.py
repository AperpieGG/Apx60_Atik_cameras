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
    Read FITS images, subtract master bias.

    Parameters
    ----------
    path : str
        Path to the image files.

    Returns
    -------
    list or None
        List of corrected images, or None if no images found.
    """

    list_images = glob.glob(os.path.join(path, '*.fits'))
    print(f'Found {len(list_images)} images in {path}.')

    if not list_images:
        print("[ERROR] No bias images found in the specified directory.")
        return None

    corrected_images = []
    for image_path in list_images:
        with fits.open(image_path, memmap=False) as hdulist:
            image_data = hdulist[0].data.astype(float)
            corrected_image = image_data
            corrected_images.append(corrected_image)

    return corrected_images


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
        with fits.open(image_path, memmap=True) as hdulist:
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


def plot_read_noise(std_row, std_col, mean_row, mean_col):
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
    plt.show()


path = '/Users/u5500483/Downloads/Bias_test/'
std_row, std_col, mean_row, mean_col = read_bias_data(path)
plot_read_noise(std_row, std_col, mean_row, mean_col)

