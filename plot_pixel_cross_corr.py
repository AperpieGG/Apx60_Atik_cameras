import argparse
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from utils_apx60 import plot_images


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


def compute_cross_correlation(image):
    """
    Compute the pixel-to-pixel cross-correlation coefficient
    as a function of spatial shift (both x and y directions).
    """
    height, width = image.shape
    max_shift = 20  # Limit the shift range for visualization

    cross_corr = np.zeros((2 * max_shift + 1, 2 * max_shift + 1))

    # Compute correlation coefficients for shifts in both x and y
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            shifted_image = np.roll(np.roll(image, dx, axis=1), dy, axis=0)
            cross_corr[dy + max_shift, dx + max_shift] = np.corrcoef(image.ravel(), shifted_image.ravel())[0, 1]

    return cross_corr


def plot_cross_correlation(cross_corr, save_path):
    """
    Plot the cross-correlation coefficient as a 3D bar plot.
    Each pixel is represented as a separate bar.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    max_shift = cross_corr.shape[0] // 2
    x = np.arange(-max_shift, max_shift + 1)
    y = np.arange(-max_shift, max_shift + 1)
    X, Y = np.meshgrid(x, y)

    # Flatten the meshgrid and correlation data
    xpos = X.ravel()
    ypos = Y.ravel()
    zpos = np.zeros_like(xpos)

    dx = dy = 1  # Bar width
    dz = cross_corr.ravel()  # Bar heights (correlation values)

    # Plot the bars
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, cmap='coolwarm')

    ax.set_xlabel('X Shift (pixels)')
    ax.set_ylabel('Y Shift (pixels)')
    ax.set_zlabel('Correlation Coefficient')
    ax.set_title('Pixel-to-Pixel Cross-Correlation')
    ax.set_zlim(0, 0.2)

    # Save figure
    save_filename = os.path.join(save_path, f'pixel_corr.png')
    plt.savefig(save_filename)
    print(f'[INFO] Read Noise plot saved: {save_filename}')

    plt.show()


# for running the cross correlation

def main():
    """
    Main function to parse command-line arguments, read bias data, and plot read noise.
    """
    plot_images()
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process bias images to calculate read noise')
    parser.add_argument('directory', type=str, help='Directory containing bias images (relative to base path).')
    args = parser.parse_args()

    plot_images()

    base_path = '/Users/u5500483/Downloads/'
    save_path = '/Users/u5500483/Downloads/'
    path = os.path.join(base_path, args.directory)

    if not os.path.exists(path):
        print(f'[ERROR] Directory {path} does not exist.')
        return

    # Compute and plot cross-correlation for the first image
    bias_values = read_images(path)
    cross_corr_matrix = compute_cross_correlation(bias_values[0])
    plot_cross_correlation(cross_corr_matrix, save_path)


if __name__ == '__main__':
    main()
