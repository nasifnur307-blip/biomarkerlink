import mne
import numpy as np

def load_chbmit_edf(path, preload=True, verbose=False):
    """
    Load a CHB-MIT .edf file with MNE and return the raw object.
    """
    raw = mne.io.read_raw_edf(path, preload=preload, verbose=verbose)
    return raw

def apply_filters(raw, l_freq=0.5, h_freq=40.0, notch_freqs=(60,)):
    """
    Apply bandpass + notch filters commonly used in EEG seizure detection.
    """
    r = raw.copy().filter(l_freq, h_freq, verbose=False)
    r.notch_filter(freqs=list(notch_freqs), verbose=False)
    return r

def segment_windows(raw, window_sec=2.0, step_sec=2.0):
    """
    Segment continuous EEG into fixed windows.
    Returns: array (n_windows, n_channels, samples_per_window)
    """
    fs = raw.info['sfreq']
    data = raw.get_data()
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    windows = []
    for start in range(0, data.shape[1] - window_samples + 1, step_samples):
        windows.append(data[:, start:start + window_samples])

    return np.stack(windows, axis=0)

def label_windows_from_annotations(raw, window_sec=2.0):
    """
    Assign seizure labels (0/1) to windows based on MNE annotations.
    """
    fs = raw.info['sfreq']
    total_samples = raw.n_times
    window_samples = int(window_sec * fs)
    n_windows = (total_samples - window_samples) // window_samples + 1
    labels = np.zeros(n_windows, dtype=int)

    for ann in raw.annotations:
        desc = ann['description'].lower()
        if 'seizure' in desc:
            onset = ann['onset']
            duration = ann['duration']
            first_win = int(np.floor(onset / window_sec))
            last_win = int(np.floor((onset + duration) / window_sec))
            labels[first_win:last_win+1] = 1

    return labels
