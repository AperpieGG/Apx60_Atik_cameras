#!/usr/bin/env python
import glob
import json
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from utils_apx60 import plot_images, CreateResiduals
import argparse

warnings.filterwarnings('ignore')
plot_images()


def load_image_data(path):
    """
    Load image data from FITS files in the specified path.
    """
    list_of_signals = glob.glob(os.path.join(path, 'ptc*.fits'))
    list_of_signals.sort()
    paired_signals = [list_of_signals[i:i + 2] for i in range(0, len(list_of_signals), 2)]
    # paired_signals = [pair for idx, pair in enumerate(paired_signals) if idx not in [0]]
    master_bias = fits.getdata(os.path.join(path, 'master_bias.fits'))
    image_1, image_2 = [], []
    exposures = []
    for pair in paired_signals:
        data_1, header_1 = fits.getdata(pair[0], header=True)
        data_2, header_2 = fits.getdata(pair[1], header=True)

        # Subtract master bias
        data_1 = data_1 - np.mean(master_bias)
        data_2 = data_2 - np.mean(master_bias)
        # Ensure data is 2D
        if data_1.ndim != 2 or data_2.ndim != 2:
            raise ValueError(f"File {pair[0]} or {pair[1]} is not 2D.")

        image_1.append(data_1)
        image_2.append(data_2)
        exposure_1 = header_1.get('EXPTIME', 'N/A')
        exposure_2 = header_2.get('EXPTIME', 'N/A')
        exposure = exposure_1 + exposure_2 / 2
        exposures.append(exposure)
        print(
            f"  Pair: {os.path.basename(pair[0])}, {os.path.basename(pair[1])} -> Exposures: {exposure_1}, {exposure_2}")

    print(f"Loaded {len(image_1)} pairs of images. Each image shape: {image_1[0].shape}")
    return np.array(image_1), np.array(image_2), np.array(exposures)


def calculate_average(image_1, image_2):
    """
    Calculate the average pixel intensity for cropped regions in two image sets.
    """
    image_1_mean = [np.mean(img) for img in image_1]
    image_2_mean = [np.mean(img) for img in image_2]
    average = np.array([(i + j) / 2 for i, j in zip(image_1_mean, image_2_mean)])
    print('Average:', average)
    return average


def calculate_variance(image_1, image_2, n):
    """
    Calculate variance for two image sets.
    """
    variance = [((i - j) - (k - l)) ** 2 / 2 * (n - 1) for i, j, k, l in
                zip(image_1, np.mean(image_1, axis=(1, 2)), image_2, np.mean(image_2, axis=(1, 2)))]
    variance = np.array([np.mean(var) for var in variance])
    print('Variance:', variance)
    return variance


def ptc_fit_high(x, a_1, b_1):
    """
    Linear fit function for PTC curve.
    """
    return a_1 * x + b_1


def fit_ptc_curve(average, variance, max_linear=64000):
    """
    Fit the PTC curve using a linear model, ignoring nonlinear regions.

    Parameters
    ----------
    average : numpy.ndarray
        Average pixel values.
    variance : numpy.ndarray
        Variance of pixel values.
    max_linear : float
        Maximum value to consider for linear fit (to exclude saturation effects).

    Returns
    -------
    tuple
        Fitted parameters and gain.
    """
    # Reject nonlinear regions
    mask = average < max_linear
    x_1 = average[mask]
    y_1 = variance[mask]

    # Perform the linear fit
    popt_1, pcov_1 = curve_fit(ptc_fit_high, x_1, y_1)
    gain = 1 / popt_1[0]

    print(f"Linear fit parameters: slope = {popt_1[0]:.6f}, intercept = {popt_1[1]:.6f}")
    print(f"Gain: {gain:.3f} e-/ADU")
    return popt_1, gain


def plot_ptc_curve(average, variance, gain, popt_1, max_linear=11800):
    """
    Plot the PTC curve.
    """
    # Identify the saturation point
    max_index = np.argmax(variance)
    saturation_grey_value = average[max_index]
    variance_sqr_saturation_grey = variance[max_index]
    fig, ax = plt.subplots()
    ax.plot(average, variance, 'ro')
    mask = (average < max_linear)
    x_fit = average[mask]
    ax.plot(x_fit, ptc_fit_high(x_fit, *popt_1), 'b-', label=f'Gain = {gain:.3f} e$^-$/ADU')
    ax.plot(saturation_grey_value, variance_sqr_saturation_grey, 'b*', label='Well Depth = %.2f ADU // %.2f e$^-$' % (
        saturation_grey_value, saturation_grey_value * gain))
    ax.set_xlabel('Pixel count (ADU)')
    ax.set_ylabel('Variance (ADU$^2$)')
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()


def save_results(average, variance, gain, popt_1, x_max, path, gain_setting,
                 linearity_error, residuals, exposures):
    """
    Save PTC results, linearity error, residuals, average values, and exposure times to a JSON file.

    Parameters:
        average (numpy.ndarray): The average pixel values.
        variance (numpy.ndarray): The variance of pixel values.
        gain (float): The calculated gain value.
        popt_1 (list): Slope and intercept from the linear fit.
        x_max (float): The saturation value.
        path (str): Directory path to save the results.
        gain_setting (int): The camera's gain setting.
        linearity_error (float): The linearity error value.
        residuals (list): Residuals from the linearity fit.
        exposures (numpy.ndarray): List of exposure times.
    """
    results = {
        'average': average.tolist(),         # Convert NumPy array to list for JSON serialization
        'variance': variance.tolist(),
        'gain': gain,
        'slope': popt_1[0],
        'intercept': popt_1[1],
        'saturation_value': x_max,
        'linearity_error': linearity_error,
        'residuals': residuals,
        'exposures': exposures.tolist()
    }

    output_file = os.path.join(path, f'ptc_results_{gain_setting}.json')
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Results saved to {output_file}")


def calculate_ptc(path, save_path, n=2):
    """
    Main function to calculate and save the PTC.
    """
    image_1, image_2, exposures = load_image_data(path)
    average = calculate_average(image_1, image_2)
    variance = calculate_variance(image_1, image_2, n)
    popt_1, gain = fit_ptc_curve(average, variance)

    max_index = np.argmax(variance)
    x_max = average[max_index]
    saturation_value = x_max

    plot_ptc_curve(average, variance, gain, popt_1)

    print('The slope for high Gain is:', popt_1)

    save_results(average, variance, gain, popt_1, x_max, save_path)
    return saturation_value, average, variance, exposures


def plot_linearity_line(gradient, offset, startx, endx, step, figure, ax1):
    plt.figure(figure)
    x_values = []
    y_values = []

    for x in np.arange(startx, endx, step):
        y = x * gradient + offset  # y = mx + c
        x_values.append(x)
        y_values.append(y)
    ax1.plot(x_values, y_values, 'b-', label='%5.3f $x$ + %5.3f' % (gradient, offset))


def plot_linearity(exposure_times, ExposureTimeList_5_95, Linearitygradient, LinearityOffset, CorrectedCtsList_5_95,
                   ResidualsList_5_95, LinearityError, ResidualsList, corrected_counts, figure, gain_setting):
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
    figure.savefig(f'Linearity_{gain_setting}.pdf', bbox_inches='tight')
    plt.show()


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


def find_linearity_error(ResidualsList):
    LinearityError = (max(ResidualsList) - min(ResidualsList)) / 2

    return LinearityError


def main():

    # Set paths
    path = '/Users/u5500483/Downloads/bias_swir/bias-default-off/flats-default-off/'
    save_path = '/Users/u5500483/Downloads/'

    # Calculate PTC
    saturation_value, average, variance, exposures = calculate_ptc(path, save_path)

    # add this point ptc is being plotted and png is saved

    figure = 2
    startx = saturation_value * 0.05
    endx = saturation_value * 0.95

    CorrectedCtsList_5_95, ExposureTimeList_5_95 = set_best_fit_ranges(average, exposures, startx, endx)
    Linearitygradient, LinearityOffset = best_fit(ExposureTimeList_5_95, CorrectedCtsList_5_95)

    range_factor = 0.9

    ResidualsList = CreateResiduals(exposures, average, LinearityOffset, Linearitygradient,
                                    saturation_value, range_factor).residuals
    ResidualsList_5_95 = CreateResiduals(ExposureTimeList_5_95, CorrectedCtsList_5_95, LinearityOffset, Linearitygradient,
                                         saturation_value, range_factor).residuals

    LinearityError = find_linearity_error(ResidualsList_5_95)

    plot_linearity(exposures, ExposureTimeList_5_95, Linearitygradient, LinearityOffset, CorrectedCtsList_5_95,
                   ResidualsList_5_95, LinearityError, ResidualsList, average, figure, gain_setting)

    print('finished!')


if __name__ == "__main__":
    main()
