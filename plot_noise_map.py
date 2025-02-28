import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from astropy.io import fits
from utils_apx60 import plot_images


def read_images(path, gain):
    """
    Reads bias images from the directory and calculates mean and standard deviation.

    Parameters:
        path (str): The directory containing bias images.
        gain (float): Gain value for the camera.

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

    bias_values = np.array(bias_values) * gain
    value_mean = np.mean(bias_values, axis=0)  # Mean per pixel
    value_std = np.std(bias_values, axis=0)  # Std deviation (noise) per pixel

    return value_mean, value_std


def plot_noise_frame(value_std, save_path):
    """
    Plots the reconstructed noise frame where each pixel represents its standard deviation.

    Parameters:
        value_std (numpy.ndarray): 2D array representing per-pixel noise (std deviation).
        save_path (str): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # trim shape of the image
    height, width = value_std.shape
    value_std = value_std[1000:height - 1000, 1000:width - 1000]
    img = ax.imshow(value_std, cmap='inferno', origin='lower')

    # Add a colorbar indicating the standard deviation values
    cbar = plt.colorbar(img, ax=ax, label='Standard Deviation (e$^-$)')

    ax.set_title(f'Pixel Noise Map')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.tight_layout()
    # Save figure
    save_filename = os.path.join(save_path, f'noise_map.pdf')
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
    parser.add_argument('gain', type=float, help='Gain value of the camera.')
    args = parser.parse_args()

    base_path = '/Users/u5500483/Downloads/'
    save_path = '/Users/u5500483/Downloads/'
    path = os.path.join(base_path, args.directory)

    if not os.path.exists(path):
        print(f'[ERROR] Directory {path} does not exist.')
        return

    # === Load Bias Data ===
    _, value_std = read_images(path, args.gain)

    # === Plot Reconstructed Noise Frame ===
    if value_std is not None:
        plot_noise_frame(value_std, save_path)
    else:
        print("[ERROR] Could not generate noise frame.")


if __name__ == "__main__":
    main()