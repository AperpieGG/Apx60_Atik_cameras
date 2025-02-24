#!/usr/bin/env python
import os
import glob
import argparse
from astropy.io import fits


def search_fits_files(directory, target_gain):
    """
    Searches for FITS files in a given directory and prints those with CAM-GAIN == target_gain.

    Parameters:
        directory (str): Path to the directory containing FITS files.
        target_gain (int): Gain value to search for in the FITS headers.
    """
    # Find all FITS files in the directory
    fits_files = glob.glob(os.path.join(directory, '*.fits'))

    if not fits_files:
        print(f"[ERROR] No FITS files found in directory: {directory}")
        return

    print(f"[INFO] Found {len(fits_files)} FITS files. Searching for CAM-GAIN = {target_gain}...")

    matching_files = []

    for fits_file in fits_files:
        try:
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                gain = header.get("CAM-GAIN")  # Read the header keyword

                if gain == target_gain:  # Check if it matches the target gain
                    print(f"[MATCH] {fits_file} (CAM-GAIN={gain})")
                    matching_files.append(fits_file)

        except Exception as e:
            print(f"[WARNING] Could not read {fits_file}: {e}")

    if not matching_files:
        print(f"[INFO] No FITS files found with CAM-GAIN = {target_gain}.")

def main():
    """
    Parses command-line arguments and executes the FITS file search.
    """
    parser = argparse.ArgumentParser(description="Search for FITS files with a specific CAM-GAIN value.")
    parser.add_argument("directory", type=str, help="Directory containing FITS files.")
    parser.add_argument("gain", type=int, help="CAM-GAIN value to search for (e.g., 62).")
    args = parser.parse_args()

    # Run the search function
    search_fits_files(args.directory, args.gain)

if __name__ == "__main__":
    main()