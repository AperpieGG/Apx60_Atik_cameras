import matplotlib.pyplot as plt
import os
import glob
import argparse
from astropy.io import fits
import numpy as np


def read_images_data(path, gain):
    """
    Reads bias images from the directory, filters by the correct gain setting,
    applies optional row banding correction, and calculates mean and standard deviation.

    Parameters:
        path (str): The directory containing bias images.
        gain (integer): The expected gain value from the camera.

    Returns:
        tuple: (mean values, standard deviation values), or (None, None) if no valid images found.
    """
    list_images = glob.glob(os.path.join(path, 'lag*.fits'))
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
    images = []
    for image_path in valid_images:
        with fits.open(image_path, memmap=True) as hdulist:
            image_data = hdulist[0].data.astype(float)

            # Trim by 100 pixels on all sides
            height, width = image_data.shape
            trimmed_image = image_data[2000:height - 2000, 2000:width - 2000]

            print(
                f"[INFO] Processing: {image_path} | Header Gain: {header_gain} | Trimmed Shape: {trimmed_image.shape}")
            images.append(trimmed_image)

    if not images:
        print("[ERROR] No valid bias images after reading. Exiting.")
        return None, None

    images = np.array(images)

    mean_images = np.mean(images, axis=0).flatten()

    return mean_images


def plot_images_vs_frame(images, save_path, gain):
    """
    Plots the average of each image vs frame number.

    Parameters:
        images (ndarray): A 2D array of images where each row is an image.
        save_path (str): The directory to save the plot.
        gain (integer): The gain setting of the camera.
    """
    fig, ax = plt.subplots()
    ax.plot(images, 'o', markersize=1)
    ax.set_xlabel('Frame Number (#)')
    ax.set_ylabel('Average value (ADU)')
    ax.set_title(f'Image Series - Gain {gain}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'lag_{gain}.png'))
    plt.close(fig)


def main():
    """Main function to parse arguments and execute PTC analysis."""
    parser = argparse.ArgumentParser(description='Test average of each image and plot vs frame number')
    parser.add_argument('gain_setting', type=int, help='Gain setting of the camera.')
    parser.add_argument('directory', type=str, help='Directory containing bias images (relative to base path).')
    args = parser.parse_args()

    base_path = '/data/'
    save_path = '/home/ops/Downloads/Lag/'
    path = os.path.join(base_path, args.directory)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    images = read_images_data(path, args.gain_setting)
    if images is None:
        return

    plot_images_vs_frame(images, save_path, args.gain_setting)

    print("[INFO] Processing complete.")


if __name__ == "__main__":
    main()

