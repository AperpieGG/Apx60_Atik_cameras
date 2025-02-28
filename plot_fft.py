import argparse
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.fft import fft
from scipy.fftpack import fft
from scipy.stats import trim_mean
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


def compute_row_fft(bias_image):
    """
    Compute the spatial power spectrum for each row in the bias image.

    Parameters:
        bias_image (numpy.ndarray): The 2D bias image.

    Returns:
        freqs (numpy.ndarray): Spatial frequency values.
        all_power_spectra (numpy.ndarray): Power spectra of all rows.
        mean_power_spectrum (numpy.ndarray): Trimmed mean power spectrum across rows.
    """
    num_rows, num_cols = bias_image.shape
    all_power_spectra = []

    for row in bias_image:
        fft_vals = fft(row)  # Apply FFT to the row
        power_spectrum = np.abs(fft_vals[:num_cols // 2]) ** 2  # Keep positive frequencies
        all_power_spectra.append(power_spectrum)

    all_power_spectra = np.array(all_power_spectra)

    # Compute the 20% trimmed mean to filter out anomalies (bad pixels, noise)
    mean_power_spectrum = trim_mean(all_power_spectra, proportiontocut=0.1, axis=0)

    freqs = np.fft.rfftfreq(num_cols)[:num_cols // 2]  # Compute frequency axis

    return freqs, all_power_spectra, mean_power_spectrum


def plot_row_power_spectrum(freqs, all_power_spectra, mean_power_spectrum, save_path, shape):
    """
    Plot the row-wise power spectra with individual spectra in grey and the average in red.

    Parameters:
        freqs (numpy.ndarray): Spatial frequency values.
        all_power_spectra (numpy.ndarray): Power spectra for each row.
        mean_power_spectrum (numpy.ndarray): Trimmed mean power spectrum.
        save_path (str): Path to save the plot.
        shape (int): Shape of the image.
    """
    plt.figure()

    # Plot individual row spectra in grey
    for spectrum in all_power_spectra:  # Only plot first 30 rows to avoid clutter
        plt.plot(freqs, spectrum, color='grey', alpha=0.5)

    # Plot the mean spectrum in red
    plt.plot(freqs, mean_power_spectrum, color='red', linewidth=2, label="Trimmed Mean Spectrum")

    plt.xlabel("Spatial Frequency")
    plt.ylabel("Power Spectrum")
    plt.title("Spatial Power Spectrum of Bias Image")
    plt.legend()
    plt.yscale('log')  # Log scale to better highlight dominant frequencies
    # Save figure
    save_filename = os.path.join(save_path, f'fft_{shape}.pdf')
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
    parser.add_argument('shape', type=int, default=2, help='Shape of the image, 1 for column and 2 for row.')
    args = parser.parse_args()

    plot_images()

    base_path = '/Users/u5500483/Downloads/'
    save_path = '/Users/u5500483/Downloads/'
    path = os.path.join(base_path, args.directory)

    if not os.path.exists(path):
        print(f'[ERROR] Directory {path} does not exist.')
        return

    # Convert list of 2D arrays to a single averaged bias image
    bias_values = read_images(path)
    if args.shape == 1:
        bias_image = np.mean(bias_values, axis=1)
    else:
        bias_image = np.mean(bias_values, axis=2)

    # Compute FFT for rows
    freqs, all_power_spectra, mean_power_spectrum = compute_row_fft(bias_image)

    # Plot the power spectrum results
    plot_row_power_spectrum(freqs, all_power_spectra, mean_power_spectrum, save_path, args.shape)


if __name__ == '__main__':
    main()
