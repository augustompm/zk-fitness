"""
feature_extraction.py

Extracts features from sensor data windows for intensity classification.

References:
    [19] Fan et al. 2014 - Feature extraction for HAR
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import MHEALTH_SAMPLING_RATE_HZ


def extract_time_domain_features(values):
    features = {}
    features["mean"] = np.mean(values)
    features["std"] = np.std(values)
    features["min"] = np.min(values)
    features["max"] = np.max(values)
    features["range"] = features["max"] - features["min"]
    features["rms"] = np.sqrt(np.mean(values ** 2))
    features["zcr"] = np.sum(np.diff(np.sign(values)) != 0) / len(values)
    return features


def extract_frequency_domain_features(values, sampling_rate=MHEALTH_SAMPLING_RATE_HZ):
    features = {}
    fft_vals = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(len(values), 1 / sampling_rate)

    features["fft_mean"] = np.mean(fft_vals)
    features["fft_max"] = np.max(fft_vals)
    features["fft_std"] = np.std(fft_vals)

    if np.sum(fft_vals) > 0:
        features["spectral_centroid"] = np.sum(freqs * fft_vals) / np.sum(fft_vals)
    else:
        features["spectral_centroid"] = 0

    return features


def extract_magnitude_features(acc_x, acc_y, acc_z):
    features = {}
    magnitude = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)

    features["magnitude_mean"] = np.mean(magnitude)
    features["magnitude_std"] = np.std(magnitude)
    features["magnitude_max"] = np.max(magnitude)
    features["sma"] = np.sum(np.abs(magnitude)) / len(magnitude)

    return features


def extract_window_features(window_data, sensor_prefix, include_frequency=True):
    all_features = {}

    acc_cols = [f"{sensor_prefix}_acc_x", f"{sensor_prefix}_acc_y", f"{sensor_prefix}_acc_z"]

    for col in acc_cols:
        if col in window_data.columns:
            values = window_data[col].values
            axis_name = col.split("_")[-1]

            time_feats = extract_time_domain_features(values)
            for feat_name, feat_val in time_feats.items():
                all_features[f"{sensor_prefix}_{axis_name}_{feat_name}"] = feat_val

            if include_frequency:
                freq_feats = extract_frequency_domain_features(values)
                for feat_name, feat_val in freq_feats.items():
                    all_features[f"{sensor_prefix}_{axis_name}_{feat_name}"] = feat_val

    if all(col in window_data.columns for col in acc_cols):
        acc_x = window_data[acc_cols[0]].values
        acc_y = window_data[acc_cols[1]].values
        acc_z = window_data[acc_cols[2]].values

        mag_feats = extract_magnitude_features(acc_x, acc_y, acc_z)
        for feat_name, feat_val in mag_feats.items():
            all_features[f"{sensor_prefix}_{feat_name}"] = feat_val

    return all_features


def extract_all_sensor_features(window_data, sensors=None, include_frequency=True):
    if sensors is None:
        sensors = ["chest", "ankle", "arm"]

    all_features = {}

    for sensor in sensors:
        sensor_features = extract_window_features(window_data, sensor, include_frequency)
        all_features.update(sensor_features)

    return all_features


def create_feature_matrix(data, window_size, sensors=None, include_frequency=True):
    n_windows = len(data) // window_size
    features_list = []
    labels = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window = data.iloc[start_idx:end_idx]

        if "label" in window.columns:
            label = window["label"].mode().values[0]
            if label == 0:
                continue
            labels.append(label)

        features = extract_all_sensor_features(window, sensors, include_frequency)
        features_list.append(features)

    return features_list, labels
