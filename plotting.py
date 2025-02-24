import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from utils_apx60 import plot_images

plot_images()


def read_rn(path):
    """
    Reads the read noise mean values from all read_noise_{gain}.json files in the given directory.

    Parameters:
        path (str): Path to the directory containing the JSON files.

    Returns:
        tuple: A dictionary where keys are the gain names (from filenames) and values are the corresponding mean read noise values.
               A sorted list of gain keys.
    """
    json_files = sorted(glob.glob(os.path.join(path, 'read_noise_*.json')))
    read_noise_values = {}
    gain_array = []

    if not json_files:
        print("[ERROR] No read_noise_{gain}.json files found in the directory.")
        return read_noise_values, gain_array

    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)

            # Extract the gain name from the filename and store it in an array
            gain = int(os.path.basename(file).replace('read_noise_', '').replace('.json', ''))
            gain_array.append(gain)

            # Extract the mean value
            mean_value = data.get('mean', None)

            if mean_value is not None:
                read_noise_values[gain] = mean_value
            else:
                print(f"[WARNING] 'mean' key missing in {file}. Skipping.")

        except Exception as e:
            print(f"[ERROR] Could not process {file}. Error: {e}")

    return read_noise_values, sorted(gain_array)  # Sort gain values


def read_ptc(path):
    """
    Reads the PTC results values from all ptc_results_{gain}.json files in the given directory.

    Parameters:
        path (str): Path to the directory containing the JSON files.

    Returns:
        dict: A dictionary where keys are the gain names and values are extracted PTC metrics.
    """
    json_files = sorted(glob.glob(os.path.join(path, 'ptc_results_*.json')))
    ptc_results = {}

    if not json_files:
        print("[ERROR] No ptc_results_{gain}.json files found in the directory.")
        return ptc_results

    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)

            # Extract the gain name from the filename
            gain = int(os.path.basename(file).replace('ptc_results_', '').replace('.json', ''))

            # Extract required values
            gain_value = data.get('gain', None)
            saturation_value = data.get('saturation_value', None)
            linearity_value = data.get('linearity_error', None)

            if gain_value is not None and saturation_value is not None and linearity_value is not None:
                ptc_results[gain] = {
                    'gain_value': gain_value,
                    'saturation_value': saturation_value * gain_value,
                    'linearity_error': linearity_value
                }
            else:
                print(f"[WARNING] Missing values in {file}. Skipping.")

        except Exception as e:
            print(f"[ERROR] Could not process {file}. Error: {e}")

    return ptc_results


def compute_read_noise_electrons(read_noise_values, ptc_results, gain_array):
    """
    Computes the read noise in electrons for each gain setting.

    Parameters:
        read_noise_values (dict): Dictionary containing read noise (ADU) for each gain.
        ptc_results (dict): Dictionary containing gain values from PTC results.
        gain_array (list): Sorted array of gain names.

    Returns:
        tuple: Arrays of read noise (electrons), gain values, saturation values (electrons), and linearity errors.
    """
    read_noise_electrons = []
    gain_values = []
    saturation_values = []
    linearity_errors = []
    dynamic_range = []

    for gain in gain_array:
        if gain in read_noise_values and gain in ptc_results:
            gain_value = ptc_results[gain]['gain_value']
            saturation_value = ptc_results[gain]['saturation_value']
            linearity_error = ptc_results[gain]['linearity_error']

            # Compute read noise in electrons
            read_noise_e = read_noise_values[gain] * gain_value

            # Compute dynamic range
            dr = 20 * np.log10(saturation_value / read_noise_e)

            # Append results
            read_noise_electrons.append(read_noise_e)
            gain_values.append(gain_value)
            saturation_values.append(saturation_value)
            linearity_errors.append(linearity_error)
            dynamic_range.append(dr)

            print(f"[INFO] Gain: {gain}, Read Noise (e-): {read_noise_e:.3f}, "
                  f"Saturation Value (e-): {saturation_value:.2f}, "
                  f"Linearity Error: {linearity_error:.3f}")

    return read_noise_electrons, gain_values, saturation_values, linearity_errors, dynamic_range


# Define paths
path_rn = '/Users/u5500483/Documents/GitHub/Apx60_Atik_cameras/RN_apx60/'
# path_rn_rr = '/Users/u5500483/Documents/GitHub/Apx60_Atik_cameras/RN_apx60_R-R/'
path_ptc = '/Users/u5500483/Documents/GitHub/Apx60_Atik_cameras/PTC_apx60/'

# Read JSON data
read_rn_values, gain_array = read_rn(path_rn)  # Now gain_array contains all extracted gain names
# read_rr_values, _ = read_rn(path_rn_rr)
read_ptc_values = read_ptc(path_ptc)

# Compute read noise in electrons and extract values
read_noise_electrons, gain_values, saturation_values, linearity_errors, dynamic_range = (
    compute_read_noise_electrons(read_rn_values, read_ptc_values, gain_array)
)

# # Compute read noise in electrons and extract values
# read_noise_rr_electrons, _, _, _, _ = (
#     compute_read_noise_electrons(read_rr_values, read_ptc_values, gain_array)
# )

# Plot everything with respect to gain
fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True, gridspec_kw={'hspace': 0})

ax[0].plot(gain_array, gain_values, 'bo-')
ax[0].grid()
ax[0].set_ylabel('Gain (e-/ADU)')

ax[1].plot(gain_array, read_noise_electrons, 'bo-')
ax[1].grid()
ax[1].set_ylabel('Read noise (e-)')

ax[2].plot(gain_array, saturation_values, 'bo-')
ax[2].grid()
ax[2].set_ylabel('FWC (e-)')

ax[3].plot(gain_array, dynamic_range, 'bo-')
ax[3].grid()
ax[3].set_ylabel('Dynamic Range (dB)')
ax[3].set_xlabel('Gain Setting')
ax[-1].xaxis.set_major_locator(ticker.MultipleLocator(10))

# ax[4].plot(gain_array, linearity_errors, 'bo-')
# ax[4].grid()
# ax[4].set_ylabel('Linearity Error')
plt.tight_layout()
# fig.savefig('Apx60.pdf', dpi=300)
plt.show()