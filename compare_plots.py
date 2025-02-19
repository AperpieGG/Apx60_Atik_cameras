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
path = '/Users/u5500483/Documents/GitHub/Apx60_Atik_cameras/'

# qhy600
path_rn_qhy = path + 'RN_qhy600/'
path_ptc_qhy = path + 'PTC_qhy600/'

# c1x
path_rn_c1x = path + 'RN_c1x/'
path_ptc_c1x = path + 'PTC_c1x/'

# apx60
path_rn_apx = path + 'RN_apx60/'
path_ptc_apx = path + 'PTC_apx60/'


# Read JSON data
rn_qhy, gain_array_qhy = read_rn(path_rn_qhy)
ptc_qhy = read_ptc(path_ptc_qhy)

rn_c1x, gain_array_c1x = read_rn(path_rn_c1x)
ptc_c1x = read_ptc(path_ptc_c1x)

rn_apx, gain_array_apx = read_rn(path_rn_apx)
ptc_apx = read_ptc(path_ptc_apx)

# Compute read noise in electrons and extract values
rn_qhy_e, gain_qhy, sat_qhy, lin_qhy, dr_qhy = compute_read_noise_electrons(rn_qhy, ptc_qhy, gain_array_qhy)
rn_c1x_e, gain_c1x, sat_c1x, lin_c1x, dr_c1x = compute_read_noise_electrons(rn_c1x, ptc_c1x, gain_array_c1x)
rn_apx_e, gain_apx, sat_apx, lin_apx, dr_apx = compute_read_noise_electrons(rn_apx, ptc_apx, gain_array_apx)

# exclude the first element of each for c1x
rn_c1x_e = rn_c1x_e[1:]
gain_c1x = gain_c1x[1:]
sat_c1x = sat_c1x[1:]
lin_c1x = lin_c1x[1:]
dr_c1x = dr_c1x[1:]


plt.figure(1)
plt.plot(gain_qhy, sat_qhy, 'bo-', label='QHY600')
plt.plot(gain_c1x, sat_c1x, 'ro-', label='C1X')
plt.plot(gain_apx, sat_apx, 'go-', label='APX60')
plt.xlabel('Gain (e-/ADU)')
plt.ylabel('Saturation Value (e-)')
plt.grid(True)
plt.legend(loc='best')
plt.savefig('/Users/u5500483/Downloads/FWC_Gain.pdf', bbox_inches='tight')
plt.show()

plt.figure(2)
plt.plot(gain_qhy, rn_qhy_e, 'bo-', label='QHY600')
plt.plot(gain_c1x, rn_c1x_e, 'ro-', label='C1X')
plt.plot(gain_apx, rn_apx_e, 'go-', label='APX60')
plt.xlabel('Gain (e-/ADU)')
plt.ylabel('Read Noise (e-)')
plt.grid(True)
plt.legend(loc='best')
plt.savefig('/Users/u5500483/Downloads/RN_Gain.pdf', bbox_inches='tight')
plt.show()


plt.figure(3)
plt.plot(gain_qhy, dr_qhy, 'bo-', label='QHY600')
plt.plot(gain_c1x, dr_c1x, 'ro-', label='C1X')
plt.plot(gain_apx, dr_apx, 'go-', label='APX60')
plt.xlabel('Gain (e-/ADU)')
plt.ylabel('Dynamic Range (DB)')
plt.grid(True)
plt.legend(loc='best')
plt.savefig('/Users/u5500483/Downloads/DR_Gain.pdf', bbox_inches='tight')
plt.show()