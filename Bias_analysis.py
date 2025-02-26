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

    list_images = glob.glob(os.path.join(path, 'image*.fits'))
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
    list_images = glob.glob(os.path.join(path, 'corr*.fits'))
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

    ax2.axhline(np.mean(mean_row), color='green', linestyle=':')

    ax2.yaxis.set_ticklabels([])

    y_min, y_max = ax1.get_ylim()
    ax2.set_ylim(y_min, y_max)

    fig.colorbar(ax1.get_children()[0], ax=ax2, label='Number of Pixels')
    fig.tight_layout()
    plt.show()


path = '/Users/u5500483/Downloads/Bias_Test/'
std_row, std_col, mean_row, mean_col = read_bias_data(path)
plot_read_noise(std_row, std_col, mean_row, mean_col)


# def compute_cross_correlation(image):
#     """
#     Compute the pixel-to-pixel cross-correlation coefficient
#     as a function of spatial shift (both x and y directions).
#     """
#     height, width = image.shape
#     max_shift = 20  # Limit the shift range for visualization
#
#     cross_corr = np.zeros((2 * max_shift + 1, 2 * max_shift + 1))
#
#     # Compute correlation coefficients for shifts in both x and y
#     for dx in range(-max_shift, max_shift + 1):
#         for dy in range(-max_shift, max_shift + 1):
#             shifted_image = np.roll(np.roll(image, dx, axis=1), dy, axis=0)
#             cross_corr[dy + max_shift, dx + max_shift] = np.corrcoef(image.ravel(), shifted_image.ravel())[0, 1]
#
#     return cross_corr
#
#
# def plot_cross_correlation(cross_corr):
#     """
#     Plot the cross-correlation coefficient as a 3D bar plot.
#     Each pixel is represented as a separate bar.
#     """
#     fig = plt.figure(figsize=(10, 6))
#     ax = fig.add_subplot(111, projection='3d')
#
#     max_shift = cross_corr.shape[0] // 2
#     x = np.arange(-max_shift, max_shift + 1)
#     y = np.arange(-max_shift, max_shift + 1)
#     X, Y = np.meshgrid(x, y)
#
#     # Flatten the meshgrid and correlation data
#     xpos = X.ravel()
#     ypos = Y.ravel()
#     zpos = np.zeros_like(xpos)
#
#     dx = dy = 1  # Bar width
#     dz = cross_corr.ravel()  # Bar heights (correlation values)
#
#     # Plot the bars
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, cmap='coolwarm')
#
#     ax.set_xlabel('X Shift (pixels)')
#     ax.set_ylabel('Y Shift (pixels)')
#     ax.set_zlabel('Correlation Coefficient')
#     ax.set_title('Pixel-to-Pixel Cross-Correlation')
#     ax.set_zlim(0, 0.2)
#     plt.show()
#
#
# def compute_power_spectrum(bias_image, axis=1):
#     """
#     Compute the spatial power spectrum for each row (or column)
#     and derive the stacked spectrum using a 20% trimmed mean.
#
#     Parameters:
#         bias_image (numpy.ndarray): The 2D bias image.
#         axis (int): Axis along which to compute the power spectrum (0 = columns, 1 = rows).
#
#     Returns:
#         freqs (numpy.ndarray): The spatial frequency values.
#         stacked_spectrum (numpy.ndarray): The trimmed mean power spectrum.
#         all_spectra (numpy.ndarray): The power spectra for each row/column.
#     """
#     if axis == 1:
#         data = bias_image  # Rows
#     else:
#         data = bias_image.T  # Columns
#
#     num_rows, num_cols = data.shape
#     power_spectra = []
#
#     for i in range(num_rows):
#         fft_vals = fft(data[i, :])  # Compute FFT for each row/column
#         power_spectrum = np.abs(fft_vals) ** 2  # Compute power spectrum
#         power_spectra.append(power_spectrum[:num_cols // 2])  # Keep only positive frequencies
#
#     power_spectra = np.array(power_spectra)
#
#     # Compute 20% trimmed mean (remove top and bottom 10% of values)
#     stacked_spectrum = trim_mean(power_spectra, proportiontocut=0.1, axis=0)
#
#     freqs = np.fft.fftfreq(num_cols)[:num_cols // 2]  # Frequency axis
#
#     return freqs, stacked_spectrum, power_spectra
#
#
# def plot_power_spectrum(freqs, stacked_spectrum, all_spectra):
#     """
#     Plot the power spectrum with individual spectra in grey and the average in red.
#
#     Parameters:
#         freqs (numpy.ndarray): The spatial frequency values.
#         stacked_spectrum (numpy.ndarray): The trimmed mean power spectrum.
#         all_spectra (numpy.ndarray): The power spectra for each row/column.
#     """
#     plt.figure(figsize=(8, 6))
#
#     # Plot individual spectra in grey
#     for spectrum in all_spectra:
#         plt.plot(freqs, spectrum, color='grey', alpha=0.5)
#
#     # Plot the trimmed mean spectrum in red
#     plt.plot(freqs, stacked_spectrum, color='red', linewidth=2, label='Trimmed Mean Spectrum')
#
#     plt.xlabel("Spatial Frequency")
#     plt.ylabel("Power Spectrum")
#     plt.title("Spatial Power Spectrum of Bias Noise")
#     plt.legend()
#     plt.yscale('log')  # Log scale for better visualization
#
#     plt.show()

# # for running the cross correlation
# def main():
#     corrected_images = read_images(path, bias(path))
#     if corrected_images is not None:
#         # Compute and plot cross-correlation for the first image
#         cross_corr_matrix = compute_cross_correlation(corrected_images[0])
#         plot_cross_correlation(cross_corr_matrix)
#
#
# if __name__ == '__main__':
#     main()

# from scipy.fftpack import fft
# from scipy.stats import trim_mean
#
#
# def compute_row_fft(bias_image):
#     """
#     Compute the spatial power spectrum for each row in the bias image.
#
#     Parameters:
#         bias_image (numpy.ndarray): The 2D bias image.
#
#     Returns:
#         freqs (numpy.ndarray): Spatial frequency values.
#         all_power_spectra (numpy.ndarray): Power spectra of all rows.
#         mean_power_spectrum (numpy.ndarray): Trimmed mean power spectrum across rows.
#     """
#     num_rows, num_cols = bias_image.shape
#     all_power_spectra = []
#
#     for row in bias_image:
#         fft_vals = fft(row)  # Apply FFT to the row
#         power_spectrum = np.abs(fft_vals[:num_cols // 2]) ** 2  # Keep positive frequencies
#         all_power_spectra.append(power_spectrum)
#
#     all_power_spectra = np.array(all_power_spectra)
#
#     # Compute the 20% trimmed mean to filter out anomalies (bad pixels, noise)
#     mean_power_spectrum = trim_mean(all_power_spectra, proportiontocut=0.1, axis=0)
#
#     freqs = np.fft.rfftfreq(num_cols)[:num_cols // 2]  # Compute frequency axis
#
#     return freqs, all_power_spectra, mean_power_spectrum
#
#
# def plot_row_power_spectrum(freqs, all_power_spectra, mean_power_spectrum):
#     """
#     Plot the row-wise power spectra with individual spectra in grey and the average in red.
#
#     Parameters:
#         freqs (numpy.ndarray): Spatial frequency values.
#         all_power_spectra (numpy.ndarray): Power spectra for each row.
#         mean_power_spectrum (numpy.ndarray): Trimmed mean power spectrum.
#     """
#     plt.figure(figsize=(10, 6))
#
#     # Plot individual row spectra in grey
#     for spectrum in all_power_spectra[:30]:  # Only plot first 30 rows to avoid clutter
#         plt.plot(freqs, spectrum, color='grey', alpha=0.5)
#
#     # Plot the mean spectrum in red
#     plt.plot(freqs, mean_power_spectrum, color='red', linewidth=2, label="Trimmed Mean Spectrum")
#
#     plt.xlabel("Spatial Frequency")
#     plt.ylabel("Power Spectrum")
#     plt.title("Spatial Power Spectrum of Bias Image Rows")
#     plt.legend()
#     plt.yscale('log')  # Log scale to better highlight dominant frequencies
#
#     plt.show()
#
#
# def main():
#     path = '/Users/u5500483/Downloads/Bias_test/'
#
#     corrected_images = read_images(path)
#     if corrected_images is None or len(corrected_images) == 0:
#         print("[ERROR] No corrected images to process. Exiting function.")
#         return
#
#     # Convert list of 2D arrays to a single averaged bias image
#     bias_image = np.mean(np.array(corrected_images), axis=2)
#
#     # Compute FFT for rows
#     freqs, all_power_spectra, mean_power_spectrum = compute_row_fft(bias_image)
#
#     # Plot the power spectrum results
#     plot_row_power_spectrum(freqs, all_power_spectra, mean_power_spectrum)
#
#
# if __name__ == '__main__':
#     main()


# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def correct_row_bias(image, reference_fraction=0.1):
#     """
#     Correct row-wise bias by identifying stable reference pixels in each row
#     and subtracting their average, then restore the global mean.
#
#     Parameters:
#         image (numpy.ndarray): 2D bias image.
#         reference_fraction (float): Fraction of pixels to use as reference.
#
#     Returns:
#         corrected_image (numpy.ndarray): Bias-corrected image with original mean restored.
#     """
#     corrected_image = np.copy(image)
#     num_rows, num_cols = image.shape
#
#     original_mean = np.mean(image)  # Store the original mean
#
#     for row_idx in range(num_rows):
#         row = image[row_idx, :]
#
#         # Determine how many pixels to use as reference
#         ref_pixel_count = max(1, int(num_cols * reference_fraction))
#
#         # Sort pixels by absolute difference from row median
#         median_val = np.median(row)
#         sorted_indices = np.argsort(np.abs(row - median_val))  # Find closest to median
#         reference_pixels = row[sorted_indices[:ref_pixel_count]]  # Select reference pixels
#
#         # Compute reference mean and subtract from entire row
#         row_correction = np.mean(reference_pixels)
#         corrected_image[row_idx, :] -= row_correction
#
#     # Restore the original mean
#     corrected_image += (original_mean - np.mean(corrected_image))
#
#     return corrected_image
#
#
# def plot_corrected_images(original, corrected):
#     """
#     Plot original and corrected bias images side by side with colorbars.
#
#     Parameters:
#         original (numpy.ndarray): Original bias image.
#         corrected (numpy.ndarray): Corrected bias image.
#     """
#     fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#
#     mean_val = np.mean(original)
#     std_val = np.std(original)
#     vmin, vmax = mean_val - std_val, mean_val + std_val
#
#     im1 = ax[0].imshow(original, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
#     ax[0].set_title("Original Bias Image")
#     fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
#
#     im2 = ax[1].imshow(corrected, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
#     ax[1].set_title("Row-Corrected Bias Image")
#     fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
#
#     plt.tight_layout()
#     plt.show()
#
#
# def main():
#     path = '/Users/u5500483/Downloads/Bias_Test/'
#     master_bias = bias(path)
#
#     if master_bias is None:
#         print("[ERROR] Master bias not created. Exiting function.")
#         return
#
#     corrected_images = read_images(path, master_bias)
#     if corrected_images is None or len(corrected_images) == 0:
#         print("[ERROR] No corrected images to process. Exiting function.")
#         return
#
#     # Convert list of 2D arrays to a single averaged bias image
#     bias_image = np.mean(np.array(corrected_images), axis=0)
#
#     # Apply row-wise correction
#     corrected_bias_image = correct_row_bias(bias_image, reference_fraction=0.05)  # Use 5% of pixels as reference
#
#     # Plot original vs corrected image
#     plot_corrected_images(bias_image, corrected_bias_image)
#
#     # Check mean value before and after correction
#     print(f"Original Mean: {np.mean(bias_image)}")
#     print(f"Corrected Mean: {np.mean(corrected_bias_image)}")
#
#
# if __name__ == '__main__':
#     main()


# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from astropy.io import fits
# import glob
#
#
# def correct_row_bias(image, reference_fraction=0):
#     """
#     Correct row-wise bias by identifying stable reference pixels in each row
#     and subtracting their average, then restore the global mean.
#
#     Parameters:
#         image (numpy.ndarray): 2D bias image.
#         reference_fraction (float): Fraction of pixels to use as reference.
#
#     Returns:
#         corrected_image (numpy.ndarray): Bias-corrected image with original mean restored.
#     """
#     corrected_image = np.copy(image)
#     num_rows, num_cols = image.shape
#
#     original_mean = np.mean(image)  # Store the original mean
#
#     for row_idx in range(num_rows):
#         row = image[row_idx, :]
#
#         # Determine how many pixels to use as reference
#         ref_pixel_count = max(1, int(num_cols * reference_fraction))
#
#         # Sort pixels by absolute difference from row median
#         median_val = np.median(row)
#         sorted_indices = np.argsort(np.abs(row - median_val))  # Find closest to median
#         reference_pixels = row[sorted_indices[:ref_pixel_count]]  # Select reference pixels
#
#         # Compute reference mean and subtract from entire row
#         row_correction = np.mean(reference_pixels)
#         corrected_image[row_idx, :] -= row_correction
#
#     # Restore the original mean
#     corrected_image += (original_mean - np.mean(corrected_image))
#
#     return corrected_image
#
#
# def save_corrected_images(path, reference_fraction=0):
#     """
#     Apply row-wise correction to all FITS images in the directory and save them
#     with 's' prefixed to the original filename.
#
#     Parameters:
#         path (str): Directory containing FITS images.
#         reference_fraction (float): Fraction of pixels used as reference in each row.
#     """
#     list_images = glob.glob(os.path.join(path, 'image*.fits'))
#     if not list_images:
#         print("[ERROR] No FITS images found in the specified directory.")
#         return
#
#     print(f"Found {len(list_images)} images to process in {path}.")
#     corrected_images = []
#     for image_path in list_images:
#         with fits.open(image_path, memmap=False) as hdulist:
#             image_data = hdulist[0].data.astype(float)
#             corrected_image = correct_row_bias(image_data, reference_fraction)
#
#             # Construct the new filename with 's' added
#             filename = os.path.basename(image_path)  # Get original filename
#             new_filename = f"s{filename}"  # Add 's' prefix
#             new_filepath = os.path.join(path, new_filename)
#
#             # Save corrected image as FITS
#             fits.writeto(new_filepath, corrected_image, hdulist[0].header, overwrite=True)
#             print(f"Saved corrected image: {new_filename}")
#
#
# def main():
#     path = '/Users/u5500483/Downloads/Bias_Test/'
#
#     # Apply correction and save corrected images
#     save_corrected_images(path, reference_fraction=0)
#
#
# if __name__ == '__main__':
#     main()
