"""
data_loader.py

Loads MHEALTH dataset for training intensity classification models.

References:
    [13] MHEALTH Dataset (Banos et al. 2014)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import (
    MHEALTH_SENSOR_COLUMNS,
    MHEALTH_ACTIVITY_TO_INTENSITY,
    INTENSITY_LABELS
)


def load_mhealth_subject(dataset_path, subject_id):
    file_path = Path(dataset_path) / f"mHealth_subject{subject_id}.log"

    if not file_path.exists():
        raise FileNotFoundError(f"Subject file not found: {file_path}")

    df = pd.read_csv(file_path, sep="\t", header=None, names=MHEALTH_SENSOR_COLUMNS)
    df["subject"] = subject_id

    return df


def load_mhealth_all_subjects(dataset_path, subjects=None):
    if subjects is None:
        subjects = range(1, 11)

    all_data = []

    for subject_id in subjects:
        try:
            df = load_mhealth_subject(dataset_path, subject_id)
            all_data.append(df)
        except FileNotFoundError:
            continue

    if not all_data:
        raise ValueError("No subject data found")

    return pd.concat(all_data, ignore_index=True)


def convert_mhealth_labels_to_intensity(labels):
    intensities = []
    for label in labels:
        intensity = MHEALTH_ACTIVITY_TO_INTENSITY.get(label)
        if intensity is not None:
            intensities.append(intensity)
        else:
            intensities.append(-1)
    return np.array(intensities)


def load_physionet_hr_session(session_path):
    hr_file = Path(session_path) / "HR.csv"

    if not hr_file.exists():
        return None

    with open(hr_file, "r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        return None

    timestamp_start = lines[0].strip()
    sampling_rate = float(lines[1].strip())

    hr_values = []
    for line in lines[2:]:
        try:
            hr_values.append(float(line.strip()))
        except ValueError:
            continue

    return {
        "timestamp_start": timestamp_start,
        "sampling_rate": sampling_rate,
        "hr_values": np.array(hr_values)
    }


def load_physionet_acc_session(session_path):
    acc_file = Path(session_path) / "ACC.csv"

    if not acc_file.exists():
        return None

    with open(acc_file, "r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        return None

    timestamp_start = lines[0].strip()
    sampling_rate = float(lines[1].strip())

    acc_values = []
    for line in lines[2:]:
        try:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                acc_values.append([float(p) for p in parts[:3]])
        except ValueError:
            continue

    return {
        "timestamp_start": timestamp_start,
        "sampling_rate": sampling_rate,
        "acc_x": np.array([v[0] for v in acc_values]),
        "acc_y": np.array([v[1] for v in acc_values]),
        "acc_z": np.array([v[2] for v in acc_values])
    }


def load_physionet_dataset(dataset_path, session_type="AEROBIC"):
    dataset_path = Path(dataset_path)
    session_dir = dataset_path / session_type

    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    sessions = []

    for subject_dir in sorted(session_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        hr_data = load_physionet_hr_session(subject_dir)
        acc_data = load_physionet_acc_session(subject_dir)

        if hr_data is not None:
            sessions.append({
                "subject": subject_dir.name,
                "session_type": session_type,
                "hr": hr_data,
                "acc": acc_data
            })

    return sessions


def get_dataset_statistics(data):
    stats = {
        "total_samples": len(data),
        "subjects": data["subject"].nunique() if "subject" in data.columns else 0,
    }

    if "label" in data.columns:
        label_counts = data["label"].value_counts().to_dict()
        stats["label_distribution"] = label_counts

    return stats
