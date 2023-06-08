import numpy as np
from scipy.signal import find_peaks, peak_widths

class OutlierDetector:
    @staticmethod
    def detect_outliers(time, flux):
        z_scores = np.abs((flux - np.mean(flux)) / np.std(flux))
        threshold = 3.0
        good_indices = np.where(z_scores <= threshold)[0]
        clean_flux = flux[good_indices]
        clean_time = time[good_indices]
        if len(clean_flux) < 0.1 * len(flux):
            clean_time = time
            clean_flux = flux
        differences = np.diff(clean_flux)
        outlier_index = np.argmax(differences) + 1
        if np.max(differences) > 0.3:
            cleaned_mag = np.delete(clean_flux, outlier_index)
            cleaned_time = np.delete(clean_time, outlier_index)
        else:
            cleaned_time = clean_time
            cleaned_mag = clean_flux
        return cleaned_time, cleaned_mag
