import pathlib
import pandas as pd
import re
import numpy as np
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import logging
from numpy.linalg import svd
from scipy.stats import norm
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import convolve1d, gaussian_filter1d, uniform_filter1d
import pathlib
STATION = "GCI006"
DATA_DIR = "/Users/matzhaugen/schumann/{station}"
POST_FILTER_WIDTH = 226
PANDEMIC_START = "2020-03-01"

LOCATIONS = {"GCI001": "California", "GCI002": "SaudiArabia", "GCI003": "Lithuania", 
             "GCI004": "Canada", "GCI005": "NewZealand", "GCI006": "SouthAfrica"}
# LOCATIONS = {"GCI004": "Canada"}

def read_data(f: pathlib.Path) -> pd.DataFrame:
    patterns = '_(\d+)_(\d+)_(\d+)'

    month = re.search(patterns, str(f)).group(2)
    day = re.search(patterns, str(f)).group(3)
    year = re.search(patterns, str(f)).group(1)
    data = pd.read_csv(f, delimiter="\t")
    data = data.rename({"-10": "freq"}, axis=1)
    pivot = data.T
    pivot = pivot.iloc[1:,:]
    hour = [f"{int(s)-1}" if int(s) > 10 else f"0{int(s)-1}" for s in pivot.index]
    datetime = [np.datetime64(f"{year}-{month}-{day} {h}:00:00") for h in hour]
    pivot.index = datetime
    return pivot

def read_freq(f) -> pd.DataFrame:
    freq = pd.read_csv(f, delimiter="\t", usecols=[0])
    return freq.values

def find_peaks2(a_hat):
    da_hat = np.diff(a_hat)
    has_swap = da_hat[:-1] * da_hat[1:] < 0
    is_max = has_swap * (da_hat[:-1] > 0)
    return np.arange(len(da_hat) - 1)[is_max] + 1


def ffill(x, na_val=0):
    a = x.copy()
    # find zero elements and associated index
    if np.isnan(na_val):
        mask = np.isnan(a)
    else:
        mask = a == na_val
    idx = np.where(~mask, np.arange(mask.size), False)

    # do the fill
    return a[np.maximum.accumulate(idx)]

def np_ffill(arr, axis):
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(np.logical_and(~np.isnan(arr), ~np.isinf(arr)), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [np.arange(k)[tuple([slice(None) if dim==i else np.newaxis
        for dim in range(len(arr.shape))])]
        for i, k in enumerate(arr.shape)]
    slc[axis] = idx
    return arr[tuple(slc)]

def get_characteristics(
    values,
    freqs,
    max_peaks=5,
    window_freq=((6.75, 9)),  # The specific frequency window to look at a peak in
    min_freq=1, # Hz  - The overall frequency range to analyse
    max_freq=30,  # - The overall frequency range to analyse
    filter_width=20, # width in peak finding filter
    post_filter_width=72, # width in filter of output
):

    sub_idx = np.logical_and(min_freq < freqs, freqs < max_freq)
    sub_freq = freqs[sub_idx]
    subset = values[:, sub_idx]
    
    subset += 0.0001
    proc_data = np.log(subset)
    n_times, n_freq = proc_data.shape

    n_windows = len(window_freq)
    final_amp = np.zeros((n_times, n_windows))
    final_freq = np.zeros((n_times, n_windows))
    final_var = np.zeros((n_times, n_windows))
    for j, window_f in enumerate(window_freq):
        
        peaks = np.zeros((n_times, max_peaks))
        main_peak = np.zeros(n_times)
        main_freq = np.zeros(n_times)
        main_var = np.zeros(n_times)
        peak_freq = np.zeros((n_times, max_peaks))
        
        for i, obs in enumerate(proc_data):
            a_hat = gaussian_filter1d(obs, filter_width)
            peaks_i = find_peaks2(a_hat)
            n_peaks = np.minimum(len(peaks_i), max_peaks)
            if n_peaks > 0:
                peaks[i, :n_peaks] = a_hat[peaks_i[:n_peaks]]
                peak_freq[i, :n_peaks] = sub_freq[peaks_i[:n_peaks]]
                main_idx = np.logical_and(window_f[0] < peak_freq[i, :], peak_freq[i, :] < window_f[1])
                if sum(main_idx) > 0:
                    main_peak[i] = peaks[i, main_idx][0]
                    main_freq[i] = peak_freq[i, main_idx][0]
                    window = np.logical_and(sub_freq > main_freq[i] - .5, sub_freq < main_freq[i] + .5)
                    main_var[i] = np.std((obs - a_hat)[window])

        final_freq[:, j] = uniform_filter1d(ffill(main_freq), post_filter_width)
        final_amp[:, j] = uniform_filter1d(ffill(main_peak), post_filter_width)
        final_var[:, j] = uniform_filter1d(ffill(main_var), post_filter_width)

    return final_amp, final_freq, final_var
         
def plot_characteristics(dates, characteristics):
    final_amp, final_freq, final_var = characteristics
    n_windows = final_amp.shape[1]
    fig, axs = plt.subplots(n_windows, 3, figsize=(35,12))

    for i in range(n_windows):
    
        axs[i, 2].plot(dates, final_var[:, i], label="Variance")
        axs[i, 0].plot(dates, final_amp[:, i], color="green", label="Amplitude")
        axs[i, 1].plot(dates, final_freq[:, i], color="red", label="Peak Freq.")   
        
        axs[i, 0].set_title(f"Amplitude from peak {i}")
        axs[i, 0].set_ylabel("Amplitude [log(pT)]")
        axs[i, 1].set_ylabel("Peak Freq [Hz]")
        axs[i, 2].set_ylabel("Standard deviation [log(pT)]")
        axs[i, 2].set_title(f"Variance from peak {i}")
        axs[i, 1].set_title(f"Peak {i} Frequency")
        fig.autofmt_xdate()  # type: ignore
        axs[i, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # type: ignore
        axs[i, 0].legend(loc="upper left")
        axs[i, 2].legend(loc="best")
        axs[i, 1].legend()
        fig.tight_layout()
        [ax.grid() for ax in axs.flatten()]
        [ax.axvline(np.datetime64(PANDEMIC_START), color="gold", alpha=0.5) for ax in axs.flatten()]
    return fig, axs

def plot_amplitude(dates, amplitude, n_days_filter=90):

    fig, axs = plt.subplots(1, 1, figsize=(15,5))
    y = amplitude
    x = np.arange(len(y))
    frac = 24*n_days_filter / len(y)
    yhat = lowess(y, x, frac=frac, return_sorted=False)
    res = np.abs(y - yhat)
    sigma_hat = np.sqrt(np.mean(np.power(res, 2)))
    # z = lowess(res / sigma_hat, x, frac=frac)[:, 1]
    z = res / sigma_hat
    plt.plot(dates, y, label="Raw amplitude")
    plt.plot(dates, yhat, label="Smooth [90-day window]", linewidth=2)
    plt.legend(loc="upper left")
    plt.title("Schuman peak amplitude")
    plt.ylabel("Amplitude [log(pT)]")
    ax = plt.gca()
    twin = ax.twinx()
    # twin.plot(dates[z>3], z[z>3], color="red", alpha=0.3)
    twin.plot(dates, z, color="green", alpha=0.5, label="Residuals")
    twin.set_yscale('log')
    twin.set_ylim([1,10])
    twin.set_ylabel("Residuals [std-normalized]")
    twin.legend(loc="lower right")
    plt.axvline(np.datetime64(PANDEMIC_START), color="gold", alpha=0.5)
    ax.text(np.datetime64("2020-03-05"), 0.98, 'Start of Pandemic - March 1st', color='black', ha='center', va='top', rotation=0,
                transform=ax.get_xaxis_transform(), fontsize=10)

    # prob = 1 / 200 * 1 / 2000
    # ax.text(np.datetime64("2020-07-01"), 0.98, f'Probability of synchrony with pandemic onset: {prob} ', color='black', ha='left', 
    #         transform=ax.get_xaxis_transform(), va='top', rotation=0, fontsize=10)

    return fig, axs

def main() -> None:

    removelist = np.concatenate([np.arange(np.datetime64('2020-04-16'), np.datetime64('2020-05-06')),
                           np.arange(np.datetime64('2019-03-18'), np.datetime64('2019-04-09'))])

    data_types = ["amp", "freq", "var"] 
    for station, location in LOCATIONS.items():
        logging.info(f"Analyzing {location}, station {station}...")

        fetched_from_scratch = False
        # if False:
        if pathlib.Path(f"proc_data/{station}{location}_freq.csv").exists():
            logging.info(f"Fetching characteristics from cache")
            characteristics = [pd.read_csv(f"proc_data/{station}{location}_{name}.csv", index_col=0) for name in data_types]
            dates = pd.to_datetime(characteristics[0].index)
            characteristics = [c.values for c in characteristics]

        else:
            files = []
            for path in pathlib.Path(DATA_DIR.format(station=station)).glob("**/*.txt"):
                if path.is_file():
                    files.append(path)

            freqs = [read_freq(f) for f in files]
            master_freq = np.copy(freqs[0])
            master_freq = master_freq.flatten()

            # Remove bad days
            bad_days = []
            for i, f in enumerate(freqs):
                if len(f) != len(master_freq):
                    bad_days.append(i)
                    
            logging.info("Importing data...")
            all_data = pd.concat([read_data(f) if i not in bad_days else pd.DataFrame() for i, f in enumerate(files)])
            logging.info("Finished importing data...")

            df = all_data.sort_index()
            
            if station == "GCI001":
                mask = ~np.in1d(df.index.date, pd.to_datetime(removelist).date)
                df = df.iloc[mask, :]

            start_date = "2019-02-01"
            end_date = "2022-12-03"
            values = df.loc[start_date:end_date].values
            dates = df.loc[start_date:end_date].index.values
            characteristics = get_characteristics(values, master_freq, post_filter_width=POST_FILTER_WIDTH, window_freq=((6.75, 9), (12, 16), (17, 22)))
            fetched_from_scratch = True

        if fetched_from_scratch:
            for name, char in zip(data_types, characteristics):
                pd.DataFrame(index=dates, data=char).to_csv(f"proc_data/{station}{location}_{name}.csv")
        
        fig, _ = plot_characteristics(dates, characteristics)
        logging.info(f" Saving to /Users/matzhaugen/schumann/SchumannAmplitude{station}.jpg")
        fig.savefig(f"/Users/matzhaugen/schumann/figures/{station}{location}Characteristics.jpg")
        plt.close(fig)
        fig, axs = plot_amplitude(dates, characteristics[0][:, 0])
        filename = f"/Users/matzhaugen/schumann/figures/{station}{location}SchumannAmplitude.jpg"
        logging.info(f" Saving to {filename}")
        fig.savefig(filename)
        plt.close(fig)
    

if __name__ == '__main__':
    logging.basicConfig(encoding='utf-8', level=logging.INFO)

    main()