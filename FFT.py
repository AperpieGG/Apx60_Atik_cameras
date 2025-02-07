#!/usr/bin/env python

import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.fft import fft, fftfreq, fftshift
from plot_images import plot_images
plot_images()
# # Define path
# path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/Bias_Dark_Frames/Bias_HDR/'
#
# # Get all FITS files
# list_images = glob.glob(path + '*.fits')
#
# # Read all images
# bias_values = []
# for image_path in list_images:
#     with fits.open(image_path, memmap=True) as hdulist:
#         image_data = hdulist[0].data.astype(np.float32)  # Convert to float32 to optimize memory
#         bias_values.append(image_data)
#
# bias_values = np.array(bias_values)  # Shape: (100, 2048, 2048)
#
# # **Step 1: Compute Row-wise Standard Deviation**
# read_noise_map = np.std(bias_values, axis=0)  # Shape: (2048, 2048)
# print(f"Read noise map computed with shape: {read_noise_map.shape}")
#
# # Save the read noise map as a FITS file (optional)
# save_path = "/Users/u5500483/Downloads/RN.fits"
# hdu = fits.PrimaryHDU(read_noise_map)
# hdu.writeto(save_path, overwrite=True)
# print(f"Read noise map saved to {save_path}")
#
# # **Step 2: Extract Row-wise Mean to Capture Banding Trends**
# row_means = np.mean(read_noise_map, axis=1)  # Averaging across columns, shape: (2048,)
#
# # **Step 3: Apply FFT to Detect Banding Frequencies**
# Fs = 1  # Spatial frequency units (1 per row)
# fft_result = fft(row_means)  # Compute FFT
# fft_magnitude = np.abs(fft_result)  # Compute amplitude spectrum
# frequencies = fftfreq(len(row_means), d=1)  # Frequency bins
#
# # Keep only positive frequencies (to remove redundancy)
# positive_freqs = frequencies[:len(frequencies) // 2]
# positive_amplitudes = fft_magnitude[:len(fft_magnitude) // 2]
#
# # **Step 4: Plot FFT Spectrum to Identify Banding Noise**
# plt.figure()
# plt.plot(positive_freqs, np.log1p(positive_amplitudes), 'b-')  # Log scale for better visualization
# plt.xlabel("Frequency [1/pixel]")
# plt.ylabel("Log Amplitude")
# plt.title("FFT Spectrum of Row-wise Standard Deviation (Detecting Banding)")
# plt.grid()
# plt.show()
#
# # **Step 5: Identify Banding Noise**
# # Find dominant frequencies (peaks in FFT spectrum)
# dominant_frequencies = positive_freqs[np.argsort(positive_amplitudes)[-5:]]  # Top 5 peak frequencies
# print(f"Dominant frequencies detected (possible banding): {dominant_frequencies}")
#
# # **Step 6: Check if Banding is Periodic or Random**
# if np.any(dominant_frequencies > 0.01):  # Arbitrary threshold for periodicity detection
#     print("⚠️ Periodic banding detected! Peaks found at nonzero frequencies.")
# else:
#     print("✅ No strong periodic banding detected. Likely random noise.")
#
#
#


#! /usr/bin/env python

# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from Read_Noise_plotting import plot_images
#
# plot_images()
#
# # Define path
# path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/Bias_Dark_Frames/Bias_HDR/'
#
# # Get all FITS files
# list_images = glob.glob(path + '*.fits')
#
# # Read all images efficiently
# bias_values = []
# for image_path in list_images:
#     with fits.open(image_path, memmap=True) as hdulist:
#         image_data = hdulist[0].data.astype(np.float32)  # Convert to float32 for efficiency
#         bias_values.append(image_data)
#
# bias_values = np.array(bias_values)  # Shape: (100, 2048, 2048)
#
# # Compute per-pixel standard deviation across all 100 images
# value_std = np.std(bias_values, axis=0)  # Shape: (2048, 2048)
#
# # Compute row-wise mean (averaging over columns)
# row_means = np.mean(value_std, axis=1)  # Shape: (2048,)
#
# # Apply FFT along rows
# fft_result = np.fft.fft(row_means)
# fft_magnitude = np.abs(fft_result)  # Amplitude spectrum
# frequencies = np.fft.fftfreq(len(row_means), d=1)  # Normalized frequency
#
# # Shift frequencies to center (for better visualization)
# fft_magnitude_shifted = np.fft.fftshift(fft_magnitude)
# frequencies_shifted = np.fft.fftshift(frequencies)
#
# # Plot FFT magnitude vs frequency
# plt.figure(figsize=(10, 5))
# plt.plot(frequencies_shifted, np.log1p(fft_magnitude_shifted), 'b-')  # Log scale for better visualization
# plt.xlabel('Frequency (1/pixels)')
# plt.ylabel('Log Amplitude')
# plt.xlim(0, 0.5)  # Limit to positive frequencies
# plt.title('FFT of Row-wise Standard Deviation (Detecting Horizontal Banding)')
# plt.grid()
# plt.show()


# #! /usr/bin/env python
#
# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits
#
# # Define path
# path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/Bias_Dark_Frames/Bias_HDR/'
#
# # Get all FITS files
# list_images = glob.glob(path + '*.fits')
# list_images = sorted(list_images[:15])
# # Read all images
# bias_values = []
# for image_path in list_images:
#     with fits.open(image_path, memmap=True) as hdulist:
#         image_data = hdulist[0].data.astype(np.float32)  # Convert to float32 to optimize memory
#         bias_values.append(image_data)
#
# bias_values = np.array(bias_values)  # Shape: (100, 2048, 2048)
# print(np.mean(np.std(bias_values, axis=0)))
# # Perform FFT along rows (axis 1)
# fft_rows = np.fft.fft(bias_values, axis=1)
# magnitude_spectrum = np.abs(fft_rows)  # Amplitude spectrum
#
# # Average over all images
# mean_spectrum = np.mean(magnitude_spectrum, axis=0)  # Shape: (2048, 2048)
#
# # Compute the mean amplitude along the columns to get a 1D profile
# mean_amplitude = np.mean(mean_spectrum, axis=1)  # Shape: (2048,)
#
# # Generate frequency axis
# num_rows = bias_values.shape[1]  # 2048 rows
# frequencies = np.fft.fftfreq(num_rows)  # Normalized frequencies
#
# # Shift frequencies to center and take log scale
# frequencies_shifted = np.fft.fftshift(frequencies)
# amplitude_shifted = np.fft.fftshift(mean_amplitude)
# log_amplitude = np.log1p(amplitude_shifted)  # Log scale for better visualization
#
# # Plot Amplitude vs Frequency
# plt.figure(figsize=(10, 5))
# plt.plot(frequencies_shifted, log_amplitude, 'b-')
# plt.xlabel('Frequency')
# plt.ylabel('Log Amplitude')
# plt.ylim(4, 7)
# plt.xlim(0, 0.52)
# plt.title('Row-wise FFT Amplitude Spectrum')
# plt.grid()
# plt.show()


# # save as above but with frequency in Hz
# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits
#
# # Define path
# path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/Bias_Dark_Frames/Bias_HDR/'
#
# # Get all FITS files
# list_images = glob.glob(path + '*.fits')
#
# # Read all images
# bias_values = []
# for image_path in list_images:
#     with fits.open(image_path, memmap=True) as hdulist:
#         image_data = hdulist[0].data.astype(np.float32)
#         bias_values.append(image_data)
#
# bias_values = np.array(bias_values)  # Shape: (100, 2048, 2048)
#
# # Define sensor parameters
# row_readout_time = 9.48e-6  # Example: 9.48 microseconds per row
# sampling_rate = 1 / row_readout_time  # Convert to Hz
#
# # Perform FFT along rows (axis 1)
# fft_rows = np.fft.fft(bias_values, axis=1)
# magnitude_spectrum = np.abs(fft_rows)
#
# # Average over all images
# mean_spectrum = np.mean(magnitude_spectrum, axis=0)  # Shape: (2048, 2048)
#
# # Compute the mean amplitude along the columns to get a 1D profile
# mean_amplitude = np.mean(mean_spectrum, axis=1)
#
# # Generate frequency axis
# num_rows = bias_values.shape[1]  # 2048 rows
# frequencies = np.fft.fftfreq(num_rows)  # Normalized frequencies
# frequencies_shifted = np.fft.fftshift(frequencies)  # Shifted for visualization
#
# # Convert to Hz
# frequencies_Hz = frequencies_shifted * sampling_rate
#
# # Apply log scaling for better visualization
# amplitude_shifted = np.fft.fftshift(mean_amplitude)
# log_amplitude = np.log1p(amplitude_shifted)
#
# # Find peak frequency
# peak_freq_index = np.argmax(amplitude_shifted)
# peak_frequency_Hz = frequencies_Hz[peak_freq_index]
# print(f"Detected interference frequency: {peak_frequency_Hz:.2f} Hz")
#
# # Plot Amplitude vs Frequency in Hz
# plt.figure(figsize=(10, 5))
# plt.plot(frequencies_Hz, log_amplitude, 'b-')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Log Amplitude')
# plt.title('Row-wise FFT Amplitude Spectrum')
# plt.ylim(0, 5)
# plt.xlim(0, np.max(frequencies_Hz))  # Show only positive frequencies
# plt.grid()
# plt.show()


import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks

# Define paths
input_path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/Bias_Dark_Frames/Bias_FFR/'
output_path = '/Users/u5500483/Downloads/Cleaned/'

# Get all FITS files
list_images = glob.glob(input_path + '*.fits')
list_images = sorted(list_images[:15])  # Process first 15 images

# Frequency setup
num_rows = 2048  # Rows in image
row_readout_time = 9.48e-6  # Example: 9.48 microseconds per row
sampling_rate = 1 / row_readout_time  # Convert to Hz
frequencies = fftfreq(num_rows, d=row_readout_time)  # Compute frequency axis
positive_freqs = frequencies[:len(frequencies) // 2]  # Keep only positive

# Read images
bias_values = []
for image_path in list_images:
    with fits.open(image_path, memmap=True) as hdulist:
        image_data = hdulist[0].data.astype(np.float32)
        bias_values.append(image_data)

bias_values = np.array(bias_values)  # Shape: (15, 2048, 2048)

# **Compute Row-wise FFT**
fft_rows = fft(bias_values, axis=1)
magnitude_spectrum = np.abs(fft_rows)

# **Average across images**
mean_spectrum = np.mean(magnitude_spectrum, axis=0)  # Shape: (2048, 2048)
mean_amplitude = np.mean(mean_spectrum, axis=1)  # Shape: (2048,)

# **Compute Adaptive Amplitude Threshold (2 × Mean)**
amplitude_threshold = 0.7 * np.mean(mean_amplitude)  # Dynamic threshold
print(mean_amplitude)
# **Find Peaks (Exclude DC Component)**
frequencies_shifted = np.fft.fftshift(frequencies)
amplitude_shifted = np.fft.fftshift(mean_amplitude)

peak_indices, _ = find_peaks(amplitude_shifted, height=amplitude_threshold)

# Ensure the frequencies and amplitudes have the same size
peak_frequencies = frequencies_shifted[peak_indices]
peak_amplitudes = amplitude_shifted[peak_indices]  # Extract corresponding amplitudes

# **Remove DC Component (0 Hz)**
valid_peaks = np.abs(peak_frequencies) > 0.1
peak_frequencies = peak_frequencies[valid_peaks]
peak_amplitudes = peak_amplitudes[valid_peaks]  # Apply same mask

print(f"Detected interference frequencies: {peak_frequencies} Hz")

# **Filter out periodic noise using a Notch Filter**
filtered_images = []
notch_width = 5  # Width around detected peaks to zero out

for i, image in enumerate(bias_values):
    fft_image = fft(image, axis=1)

    # Remove detected peaks (notch filter)
    for peak_freq in peak_frequencies:
        freq_indices = np.where(np.abs(frequencies - peak_freq) < notch_width * row_readout_time)
        fft_image[:, freq_indices] = 0  # Zero out those frequencies

    # Inverse FFT to reconstruct the image
    filtered_image = np.real(ifft(fft_image, axis=1))
    filtered_images.append(filtered_image)

    # Save new filtered image
    output_filename = output_path + list_images[i].split('/')[-1].replace('.fits', '_filtered.fits')
    hdu = fits.PrimaryHDU(filtered_image)
    hdu.writeto(output_filename, overwrite=True)
    print(f"Saved filtered image: {output_filename}")

filtered_images = np.array(filtered_images)

# **Compare Read Noise (Standard Deviation) Before & After**
std_before = np.std(bias_values)
std_after = np.std(filtered_images)
print(f"Read noise before filtering: {std_before:.4f} ADU")
print(f"Read noise after filtering: {std_after:.4f} ADU")

# **Plot FFT Spectrum with Detected Peaks**
plt.figure(figsize=(10, 5))
plt.plot(frequencies_shifted, np.log1p(amplitude_shifted), 'b-', label="FFT Amplitude")
plt.scatter(peak_frequencies, np.log1p(peak_amplitudes), color='r', label="Detected Peaks")  # FIXED
plt.axhline(np.log1p(amplitude_threshold), color="g", linestyle="--", label="Threshold")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Log Amplitude")
plt.ylim(4, 8)
plt.xlim(0, np.max(frequencies_shifted))  # Limit to positive frequencies
plt.title("Row-wise FFT Amplitude Spectrum with Peak Detection")
plt.legend()
plt.grid()
plt.show()
