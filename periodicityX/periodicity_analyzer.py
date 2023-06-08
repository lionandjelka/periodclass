import numpy as np
from scipy.signal import find_peaks, peak_widths
from periodicity.utils.correlation import correlation_nd

class PeriodicityAnalyzer:
    def __init__(self):
        self.fs_gp = None

    def set_fs_gp(self, fs_gp):
        self.fs_gp = fs_gp

    def get_qso(self, set11):
        sett = []
        for set1 in range(len(set11)):
            demo_lc = self.fs_gp.get_group(str(set11[set1]))
            d0 = demo_lc[demo_lc['filter'] == 1].sort_values(by=['mjd']).dropna()
            d1 = demo_lc[demo_lc['filter'] == 2].sort_values(by=['mjd']).dropna()
            d2 = demo_lc[demo_lc['filter'] == 3].sort_values(by=['mjd']).dropna()
            d3 = demo_lc[demo_lc['filter'] == 4].sort_values(by=['mjd']).dropna()
            d4 = demo_lc[demo_lc['filter'] == 5].sort_values(by=['mjd']).dropna()
            if (len(d0) >= 100) and (len(d1) >= 100) and (len(d2) > 100) and (len(d3) >= 100):
                sett.append(str(set11[set1]))
        return sett

    def get_lc22(self, set1):
        demo_lc = self.fs_gp.get_group(set1)
        d0 = demo_lc[(demo_lc['filter'] == 1)].sort_values(by=['mjd']).dropna()
        d1 = demo_lc[(demo_lc['filter'] == 2)].sort_values(by=['mjd']).dropna()
        d2 = demo_lc[(demo_lc['filter'] == 3)].sort_values(by=['mjd']).dropna()
        d3 = demo_lc[(demo_lc['filter'] == 4)].sort_values(by=['mjd']).dropna()
        d4 = demo_lc[(demo_lc['filter'] == 5)].sort_values(by=['mjd']).dropna()
        if (len(d0) < 100) or (len(d1) < 100) or (len(d2) < 100) or (len(d3) < 100):
            return
        x = np.array([d0['mjd'], d1['mjd'], d2['mjd'], d3['mjd']]).T
        lc = np.array([d0['flux'], d1['flux'], d2['flux'], d3['flux']]).T
        lc_err = np.array([d0['flux_err'], d1['flux_err'], d2['flux_err'], d3['flux_err']]).T
        return x, lc, lc_err

    def do_search(self, periodic_objects):
        df_full = pd.DataFrame()
        print('Total objects:', len(periodic_objects))
        for i, obj in enumerate(periodic_objects):
            print(f'Processing object {i + 1}/{len(periodic_objects)}:', obj)
            result = self.search_single(obj)
            if result is not None:
                df_full = pd.concat([df_full, result], axis=0)
        return df_full

    def search_single(self, obj):
        result_df = pd.DataFrame()
        time, flux, flux_err = self.get_lc22(obj)
        if time is None:
            return None
        cleaned_time, cleaned_flux = OutlierDetector.detect_outliers(time, flux)
        if len(cleaned_time) < 50:
            return None
        peaks, properties = find_peaks(cleaned_flux, distance=10, prominence=0.1)
        prominences = properties["prominences"]
        widths_half = peak_widths(cleaned_flux, peaks, rel_height=0.5)[0]
        widths_full = peak_widths(cleaned_flux, peaks, rel_height=1)[0]
        if len(peaks) == 0:
            return None
        if len(peaks) > 3:
            mean_prominence = np.mean(prominences)
            std_prominence = np.std(prominences)
            width_cond = np.logical_and(widths_half >= 10, widths_half <= 200)
            prom_cond = prominences >= mean_prominence + 3 * std_prominence
            selected_peaks = peaks[np.logical_and(width_cond, prom_cond)]
        else:
            selected_peaks = peaks
        for i, idx in enumerate(selected_peaks):
            peak_time = cleaned_time[idx]
            peak_flux = cleaned_flux[idx]
            period, best_period, score = self.get_period(cleaned_time, cleaned_flux, idx)
            if period is not None:
                entry = pd.DataFrame({"objectId": [obj],
                                      "period": [period],
                                      "best_period": [best_period],
                                      "score": [score],
                                      "peak_time": [peak_time],
                                      "peak_flux": [peak_flux]})
                result_df = pd.concat([result_df, entry], axis=0)
        return result_df

    def get_period(self, time, flux, peak_idx):
        peak_time = time[peak_idx]
        peak_flux = flux[peak_idx]
        shift_flux = flux - peak_flux
        correlation = correlation_nd(shift_flux, shift_flux)
        correlation_time = time - peak_time
        correlation = correlation[np.argwhere(correlation_time >= 0).flatten()]
        correlation_time = correlation_time[np.argwhere(correlation_time >= 0).flatten()]
        best_period, period_score = self.find_best_period(correlation_time, correlation)
        period = None
        score = None
        if best_period is not None:
            score = period_score
            period = best_period
        return period, best_period, score

    @staticmethod
    def find_best_period(time, correlation):
        period, power = periodogram(time, correlation)
        period_score = None
        best_period = None
        if len(period) > 0:
            max_power_idx = np.argmax(power)
            best_period = period[max_power_idx]
            period_score = power[max_power_idx]
        return best_period, period_score
