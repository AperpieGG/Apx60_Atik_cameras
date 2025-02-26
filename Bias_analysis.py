import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from utils_apx60 import plot_images

plot_images()

def read_master_bias(path):
    """
    Read master bias image.

    Parameters:
        path (str): Path to the master bias image.

    Returns:
        numpy.ndarray: 2D array representing the master bias image.
    """
    master_bias_path = os.path.join(path, 'master_bias.fits')
    if not os.path.exists(master_bias_path):
        print("[ERROR] Master bias image not found.")
        return None

    with fits.open(master_bias_path, memmap=False) as hdulist:
        master_bias = hdulist[0].data.astype(float)

    return master_bias


def read_images(path, master_bias):
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

    list_images = glob.glob(os.path.join(path, 'image*.fits'))
    print(f'Found {len(list_images)} images in {path}.')

    if not list_images:
        print("[ERROR] No bias images found in the specified directory.")
        return None

    corrected_images = []
    for image_path in list_images:
        with fits.open(image_path, memmap=False) as hdulist:
            image_data = hdulist[0].data.astype(float)
            corrected_image = image_data - master_bias
            corrected_images.append(corrected_image)

    return corrected_images


def plot_corrected_images(original, corrected):
    """
    Plot original and corrected bias images side by side with colorbars.

    Parameters:
        original (numpy.ndarray): Original bias image.
        corrected (numpy.ndarray): Corrected bias image.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    mean_val = np.mean(corrected)
    std_val = np.std(corrected)
    vmin, vmax = mean_val - 2 * std_val, mean_val + 2 * std_val
    print(vmin, vmax)

    im1 = ax[0].imshow(original, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    ax[0].set_title("Original Bias Image")
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)

    im2 = ax[1].imshow(corrected, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    ax[1].set_title("Row-Corrected Bias Image")
    fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def correct_row_bias(image, reference_fraction):
    """
    Correct row-wise bias by identifying stable reference pixels in each row
    and subtracting their average, then restore the global mean.

    Parameters:
        image (numpy.ndarray): 2D bias image.
        reference_fraction (float): Fraction of pixels to use as reference.

    Returns:
        corrected_image (numpy.ndarray): Bias-corrected image with original mean restored.
    """
    corrected_image = np.copy(image)
    num_rows, num_cols = image.shape

    original_mean = np.mean(image)  # Store the original mean

    for row_idx in range(num_rows):
        row = image[row_idx, :]

        # Determine how many pixels to use as reference
        ref_pixel_count = max(1, int(num_cols * reference_fraction))

        # Sort pixels by absolute difference from row median
        median_val = np.median(row)
        sorted_indices = np.argsort(np.abs(row - median_val))  # Find closest to median
        reference_pixels = row[sorted_indices[:ref_pixel_count]]  # Select reference pixels

        # Compute reference mean and subtract from entire row
        row_correction = np.mean(reference_pixels)
        corrected_image[row_idx, :] -= row_correction

    # Restore the original mean
    corrected_image += (original_mean - np.mean(corrected_image))

    return corrected_image


def save_corrected_images(path, reference_fraction):
    """
    Apply row-wise correction to all FITS images in the directory and save them
    with 's' prefixed to the original filename.

    Parameters:
        path (str): Directory containing FITS images.
        reference_fraction (float): Fraction of pixels used as reference in each row.
    """
    list_images = glob.glob(os.path.join(path, 'image*.fits'))
    if not list_images:
        print("[ERROR] No FITS images found in the specified directory.")
        return

    print(f"Found {len(list_images)} images to process in {path}.")
    corrected_images = []
    for image_path in list_images:
        with fits.open(image_path, memmap=False) as hdulist:
            image_data = hdulist[0].data.astype(float)
            corrected_image = correct_row_bias(image_data, reference_fraction)

            # Construct the new filename with 's' added
            filename = os.path.basename(image_path)  # Get original filename
            new_filename = f"s{filename}"  # Add 's' prefix
            new_filepath = os.path.join(path, new_filename)
            # if already saved then skip
            if os.path.exists(new_filepath):
                print(f"Already saved corrected image: {new_filename}")
                continue
            # Save corrected image as FITS
            fits.writeto(new_filepath, corrected_image, hdulist[0].header, overwrite=True)
            # print(f"Saved corrected image: {new_filename}")


def main():
    path = '/Users/u5500483/Downloads/Bias_Test/'

    images = read_images(path, master_bias=read_master_bias(path))
    if images is None or len(images) == 0:
        print("[ERROR] No images to process. Exiting function.")
        return

    # Convert list of 2D arrays to a single averaged bias image
    bias_image = np.mean(np.array(images), axis=0)
    reference_fraction = 0.05
    # Apply row-wise correction
    corrected_bias_image = correct_row_bias(bias_image, reference_fraction)

    # Plot original vs corrected image
    plot_corrected_images(images[0], corrected_bias_image)

    # Check mean value before and after correction
    print(f"Original Mean: {np.mean(images[0])}")
    print(f"Corrected Mean: {np.mean(corrected_bias_image)}")
    # Apply correction and save corrected images
    save_corrected_images(path, reference_fraction)


if __name__ == '__main__':
    main()
