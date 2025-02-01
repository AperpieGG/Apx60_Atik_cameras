import os
import json
import glob


def read_rn(path):
    """
    Reads the read noise mean values from all read_noise_{gain}.json files in the given directory.

    Parameters:
        path (str): Path to the directory containing the JSON files.

    Returns:
        dict: A dictionary where keys are the gain values and values are the corresponding mean read noise values.
    """
    json_files = glob.glob(os.path.join(path, 'read_noise_*.json'))
    read_noise_values = {}

    if not json_files:
        print("[ERROR] No read_noise_{gain}.json files found in the directory.")
        return read_noise_values

    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)

            # Extract the gain value from the filename
            gain = os.path.basename(file).replace('read_noise_', '').replace('.json', '')

            # Extract the mean value
            mean_value = data.get('mean', None)

            if mean_value is not None:
                read_noise_values[gain] = mean_value
                print(f"[INFO] Gain: {gain}, Mean Read Noise: {mean_value}")
            else:
                print(f"[WARNING] 'mean' key missing in {file}. Skipping.")

        except Exception as e:
            print(f"[ERROR] Could not process {file}. Error: {e}")

    return read_noise_values


# Example usage:
path = '/home/ops/Downloads/RN/'
read_rn_values = read_rn(path)
