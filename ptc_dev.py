#!/usr/bin/env python
import glob
import json
import os
import warnings
import argparse
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from utils_apx60 import plot_images, CreateResiduals

warnings.filterwarnings('ignore')
plot_images()


def load_image_data(path, gain_setting):
    """
    Load image data from FITS files in the specified path while verifying gain settings
    and ensuring the images in each pair have the same exposure time.

    Parameters:
        path (str): Directory containing PTC images.
        gain_setting (int): Expected gain setting.

    Returns:
        tuple: Arrays of trimmed images (image_1, image_2) and exposure times.
    """
    list_of_signals = glob.glob(os.path.join(path, '*.fits'))
    list_of_signals.sort()

    if len(list_of_signals) < 2:
        raise ValueError("Not enough images to create a master bias.")

    # Find two bias images with EXPTIME = 0.001s and correct CAM-GAIN
    bias_images = []
    for image_path in list_of_signals:
        with fits.open(image_path) as hdulist:
            header_gain = hdulist[0].header.get('CAM-GAIN', None)
            exposure_time = hdulist[0].header.get('EXPTIME', None)

            if header_gain == gain_setting and exposure_time == 0.001:
                bias_images.append(image_path)

            if len(bias_images) == 2:
                break  # Stop when two valid bias images are found

    if len(bias_images) < 2:
        raise ValueError("Could not find two bias images with EXPTIME = 0.001s and the specified gain setting.")

    # Load bias images
    bias_1, _ = fits.getdata(bias_images[0], header=True)
    bias_2, _ = fits.getdata(bias_images[1], header=True)

    # Compute master bias
    master_bias = (bias_1.astype(float) + bias_2.astype(float)) / 2
    print(f"Master bias computed from {bias_images[0]} and {bias_images[1]}.")

    # Process remaining images that match the specified gain
    filtered_signals = []
    for image_path in list_of_signals:
        if image_path in bias_images:
            continue  # Skip bias images

        with fits.open(image_path) as hdulist:
            header_gain = hdulist[0].header.get('CAM-GAIN', None)

            if header_gain == gain_setting:
                filtered_signals.append(image_path)
            else:
                print(f"Skipping {image_path} due to mismatched gain (Header: {header_gain}, Expected: {gain_setting}).")

    # Ensure valid pairs of images
    if len(filtered_signals) < 2:
        raise ValueError("Not enough valid images with the specified gain setting.")

    paired_signals = [filtered_signals[i:i + 2] for i in range(0, len(filtered_signals), 2)]
    image_1, image_2, exposures = [], [], []

    for pair in paired_signals:
        data_1, header_1 = fits.getdata(pair[0], header=True)
        data_2, header_2 = fits.getdata(pair[1], header=True)

        # Extract exposure times and ensure they are identical for the pair
        exposure_1 = header_1.get('EXPTIME', None)
        exposure_2 = header_2.get('EXPTIME', None)

        if exposure_1 is None or exposure_2 is None or exposure_1 != exposure_2:
            print(f"Skipping {pair[0]} and {pair[1]} due to mismatched exposure times.")
            continue

        # Subtract master bias
        data_1 = data_1.astype(float) - master_bias
        data_2 = data_2.astype(float) - master_bias

        # Ensure data is 2D
        if data_1.ndim != 2 or data_2.ndim != 2:
            raise ValueError(f"File {pair[0]} or {pair[1]} is not 2D.")

        # Trim the image by 1000 pixels from each side
        height, width = data_1.shape
        trimmed_1 = data_1[2000:height - 2000, 1000:width - 2000]
        trimmed_2 = data_2[2000:height - 2000, 1000:width - 2000]

        image_1.append(trimmed_1)
        image_2.append(trimmed_2)
        exposures.append(exposure_1)  # Store only one since they are identical

    print(f"Loaded {len(image_1)} valid pairs of images with gain {gain_setting}.")
    return np.array(image_1), np.array(image_2), np.array(exposures)


def calculate_average(image_1, image_2):
    """Calculate the average pixel intensity from two image sets."""
    return np.array([(np.mean(img1) + np.mean(img2)) / 2 for img1, img2 in zip(image_1, image_2)])


def calculate_variance(image_1, image_2, n=2):
    """
    Calculate variance for two image sets.
    """
    variance = [((i - j) - (k - l)) ** 2 / 2 * (n - 1) for i, j, k, l in
                zip(image_1, np.mean(image_1, axis=(1, 2)), image_2, np.mean(image_2, axis=(1, 2)))]
    variance = np.array([np.mean(var) for var in variance])
    return variance


def ptc_fit_high(x, a, b):
    """Linear function for PTC fitting."""
    return a * x + b


def fit_ptc_curve(average, variance, max_linear=40000):
    """Fit a linear PTC curve while excluding nonlinear regions."""
    mask = average < max_linear
    popt, _ = curve_fit(ptc_fit_high, average[mask], variance[mask])
    gain = 1 / popt[0]
    return popt, gain


def plot_ptc_curve(average, variance, gain, popt, save_path, gain_setting, max_linear=40000):
    """Plot and save the PTC curve."""
    fig, ax = plt.subplots()
    max_index = np.argmax(variance)
    saturation_grey_value = average[max_index]
    variance_sqr_saturation_grey = variance[max_index]
    ax.plot(average, variance, 'ro')
    mask = (average < max_linear)
    x_fit = average[mask]
    ax.plot(x_fit, ptc_fit_high(x_fit, *popt), 'b-', label=f'Gain = {gain:.3f} e$^-$/ADU')
    ax.plot(saturation_grey_value, variance_sqr_saturation_grey, 'b*', label='FWC = %.1f ADU // %.1f e$^-$' % (
        saturation_grey_value, saturation_grey_value * gain))

    ax.set_xlabel('Pixel count (ADU)')
    ax.set_ylabel('Variance (ADU$^2$)')
    ax.set_title(f'PTC - Gain {gain_setting}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'ptc_curve_{gain_setting}.png'))
    plt.close(fig)


def save_results(average, variance, gain, popt, saturation_value, path, gain_setting, linearity_error, residuals,
                 exposures):
    """Save PTC analysis results to a JSON file."""
    results = {
        'average': average.tolist(),
        'variance': variance.tolist(),
        'gain': gain,
        'slope': popt[0],
        'intercept': popt[1],
        'saturation_value': saturation_value,
        'linearity_error': linearity_error,
        'residuals': residuals,
        'exposures': exposures.tolist()
    }

    with open(os.path.join(path, f'ptc_results_{gain_setting}.json'), 'w') as file:
        json.dump(results, file, indent=4)
    print(f"Results saved: {path}ptc_results_{gain_setting}.json")


def calculate_ptc(path, save_path, gain_setting):
    """Calculate and save PTC results."""
    image_1, image_2, exposures = load_image_data(path, gain_setting)
    average = calculate_average(image_1, image_2)
    variance = calculate_variance(image_1, image_2, n=2)
    popt, gain = fit_ptc_curve(average, variance)

    saturation_value = average[np.argmax(variance)]
    plot_ptc_curve(average, variance, gain, popt, save_path, gain_setting)

    return saturation_value, average, variance, exposures, gain, popt


def find_linearity_error(residuals):
    """Calculate linearity error."""
    return (max(residuals) - min(residuals)) / 2


def plot_linearity_line(gradient, offset, startx, endx, step, figure, ax1):
    plt.figure(figure)
    x_values = []
    y_values = []

    for x in np.arange(startx, endx, step):
        y = x * gradient + offset  # y = mx + c
        x_values.append(x)
        y_values.append(y)
    ax1.plot(x_values, y_values, 'b-', label='%5.3f $x$ + %5.3f' % (gradient, offset))


def plot_linearity(save_path, exposure_times, ExposureTimeList_5_95, Linearitygradient, LinearityOffset, CorrectedCtsList_5_95,
                   ResidualsList_5_95, LinearityError, ResidualsList, corrected_counts, gain_setting):
    startx = (min(ExposureTimeList_5_95))
    endx = (max(ExposureTimeList_5_95))
    step = 0.0001

    figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.set_ylabel('Signal [cts]')
    ax1.plot([exposure_times], [corrected_counts], 'ro')
    plot_linearity_line(Linearitygradient, LinearityOffset, startx, endx, step, figure, ax1)
    ax1.axvline(x=startx, color='b', linestyle='--', linewidth=1)
    ax1.axvline(x=endx, color='b', linestyle='--', linewidth=1)
    ax1.legend(loc='best')

    ax2.plot(exposure_times, ResidualsList, 'ro', linewidth=1, label=' LE = $\\pm$ %5.3f %%' % LinearityError)
    ax2.plot([startx, endx], [0, 0], 'b-', linewidth=1)
    ax2.set_ylim(-3 * LinearityError, +3 * LinearityError)
    ax2.set_ylabel('Residuals [%]')
    ax2.set_xlabel('Exposure [s]')
    ax2.axvline(x=startx, color='b', linestyle='--', linewidth=1)
    ax2.axvline(x=endx, color='b', linestyle='--', linewidth=1)
    ax2.legend(loc='best')
    plt.tight_layout()
    figure.savefig(os.path.join(save_path, f'Linearity_{gain_setting}.png'))
    plt.close(figure)


def set_best_fit_ranges(xdata, ydata, startx, endx):
    best_fit_xdata = []
    best_fit_ydata = []

    for x, y in zip(xdata, ydata):
        if startx < x < endx:
            best_fit_xdata.append(x)
            best_fit_ydata.append(y)

    return best_fit_xdata, best_fit_ydata


def best_fit(xdata, ydata):
    def func(x, a, b):
        return a * x + b

    Gradient = curve_fit(func, xdata, ydata)[0][0]
    Offset = curve_fit(func, xdata, ydata)[0][1]

    print('gradient [{}] offset [{}]'.format(Gradient, Offset))
    return Gradient, Offset


def main():
    """Main function to parse arguments and execute PTC analysis."""
    parser = argparse.ArgumentParser(description='Photon Transfer Curve (PTC) Analysis')
    parser.add_argument('gain_setting', type=int, help='Gain setting of the camera.')
    parser.add_argument('directory', type=str, help='Directory containing bias images (relative to base path).')
    args = parser.parse_args()

    base_path = '/data/'
    save_path = '/home/ops/Downloads/PTC/'
    path = os.path.join(base_path, args.directory)

    # Compute PTC
    saturation_value, average, variance, exposures, gain, popt = calculate_ptc(path, save_path,
                                                                               args.gain_setting)

    # **Apply 5%-95% range of the saturation value**
    startx = saturation_value * 0.05
    endx = saturation_value * 0.95

    CorrectedCtsList_5_95, ExposureTimeList_5_95 = set_best_fit_ranges(average, exposures, startx, endx)
    Linearitygradient, LinearityOffset = best_fit(ExposureTimeList_5_95, CorrectedCtsList_5_95)

    range_factor = 0.9

    ResidualsList = CreateResiduals(exposures, average, LinearityOffset, Linearitygradient,
                                    saturation_value, range_factor).residuals
    ResidualsList_5_95 = CreateResiduals(ExposureTimeList_5_95, CorrectedCtsList_5_95, LinearityOffset,
                                         Linearitygradient,
                                         saturation_value, range_factor).residuals

    LinearityError = find_linearity_error(ResidualsList_5_95)

    plot_linearity(save_path, exposures, ExposureTimeList_5_95, Linearitygradient, LinearityOffset,
                   CorrectedCtsList_5_95,
                   ResidualsList_5_95, LinearityError, ResidualsList, average, args.gain_setting)

    save_results(average, variance, gain, popt, saturation_value, save_path, args.gain_setting, LinearityError,
                 ResidualsList, exposures)


if __name__ == "__main__":
    main()
