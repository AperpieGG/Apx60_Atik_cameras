import matplotlib.pyplot as plt
import os
import glob
import argparse
from astropy.io import fits
import numpy as np


def read_images_data(path, gain):
    """
    Reads bias images from the directory, filters by the correct gain setting,
    applies optional row banding correction, and calculates mean per image.

    Parameters:
        path (str): The directory containing bias images.
        gain (integer): The expected gain value from the camera.

    Returns:
        list: A list of mean values for each frame, or None if no valid images are found.
    """
    list_images = glob.glob(os.path.join(path, 'lag*.fits'))
    print(f'Found {len(list_images)} images in {path}.')

    if not list_images:
        print(f"[ERROR] No bias images found in {path}.")
        return None

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
        return None

    # **Process each valid image and compute mean per frame**
    mean_per_frame = []
    for image_path in valid_images:
        with fits.open(image_path, memmap=True) as hdulist:
            image_data = hdulist[0].data.astype(float)

            # Trim by 100 pixels on all sides
            height, width = image_data.shape
            trimmed_image = image_data[2000:height - 2000, 2000:width - 2000]

            frame_mean = np.mean(trimmed_image)  # Compute mean per image
            mean_per_frame.append(frame_mean)

            print(f"[INFO] Processed: {image_path} | Mean Value: {frame_mean:.2f}")

    return mean_per_frame


def plot_images_vs_frame(mean_values, save_path, gain):
    """
    Plots the average value of each image vs frame number.

    Parameters:
        mean_values (list): A list of average pixel values per frame.
        save_path (str): The directory to save the plot.
        gain (integer): The gain setting of the camera.
    """
    frame_numbers = np.arange(1, len(mean_values) + 1)  # Generate frame numbers

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(frame_numbers, mean_values, 'o-', markersize=4, label="Frame Mean")
    ax.set_xlabel('Frame Number (#)')
    ax.set_ylabel('Average Value (ADU)')
    ax.set_title(f'Average Image Value vs Frame Number (Gain {gain})')
    plt.tight_layout()

    # Save the figure
    plot_filename = os.path.join(save_path, f'lag_{gain}.png')
    plt.savefig(plot_filename)
    print(f"[INFO] Plot saved: {plot_filename}")
    plt.show()


def main():
    """Main function to parse arguments and execute the script."""
    parser = argparse.ArgumentParser(description='Compute average of each image and plot vs frame number')
    parser.add_argument('gain_setting', type=int, help='Gain setting of the camera.')
    parser.add_argument('directory', type=str, help='Directory containing bias images (relative to base path).')
    args = parser.parse_args()

    base_path = '/data/'
    save_path = '/home/ops/Downloads/Lag/'
    path = os.path.join(base_path, args.directory)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mean_values = read_images_data(path, args.gain_setting)
    if mean_values is None:
        return

    plot_images_vs_frame(mean_values, save_path, args.gain_setting)

    print("[INFO] Processing complete.")


if __name__ == "__main__":
    main()