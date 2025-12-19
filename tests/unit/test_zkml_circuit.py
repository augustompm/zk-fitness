"""
test_zkml_circuit.py

Unit tests for the zkML fitness intensity classification circuit.
Tests the embedded Decision Tree, timestamp validation, duration calculation,
and WHO 150 equivalent minutes goal verification.

References:
    [1] WHO Physical Activity Guidelines 2020
    [12] ACSM Guidelines 10th ed
    [21] Breiman et al. 1984 - CART
"""

import pytest
import subprocess
import json
import os
import tempfile
from pathlib import Path


CIRCUIT_DIR = Path(__file__).parent.parent.parent / "circuits"
BUILD_DIR = CIRCUIT_DIR / "build"
WASM_FILE = BUILD_DIR / "zkml_fitness_classifier_js" / "zkml_fitness_classifier.wasm"
WITNESS_GEN = BUILD_DIR / "zkml_fitness_classifier_js" / "generate_witness.js"
ZKEY_FILE = BUILD_DIR / "zkml_fitness_classifier.zkey"

WHO_GOAL_EQUIVALENT_MINUTES = 150
MAX_TIMESTAMP_GAP_SECONDS = 60
VIGOROUS_MULTIPLIER = 2

INTENSITY_REST = 0
INTENSITY_MODERATE = 1
INTENSITY_VIGOROUS = 2

READINGS_PER_SESSION = 12
N_SESSIONS = 10


def create_session_timestamps(base_time, num_readings, gap_seconds):
    return [str(base_time + i * gap_seconds) for i in range(num_readings)]


def create_rest_features():
    return ["980", "45", "1100", "50", "85", "950", "20", "90"]


def create_moderate_features():
    return ["980", "350", "1300", "95", "115", "1020", "55", "110"]


def create_vigorous_features():
    return ["980", "600", "1300", "120", "145", "1100", "65", "160"]


def generate_witness(input_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(input_data, f)
        input_file = f.name

    witness_file = tempfile.NamedTemporaryFile(suffix='.wtns', delete=False).name

    try:
        result = subprocess.run(
            ['node', str(WITNESS_GEN), str(WASM_FILE), input_file, witness_file],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return None, result.stderr

        return witness_file, None
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    finally:
        os.unlink(input_file)


def get_public_signals(witness_file):
    try:
        proof_file = str(Path(witness_file).parent / "temp_proof.json")
        public_file = str(Path(witness_file).parent / "temp_public.json")

        cmd = f'npx snarkjs groth16 prove "{ZKEY_FILE}" "{witness_file}" "{proof_file}" "{public_file}"'
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BUILD_DIR),
            timeout=60,
            shell=True
        )

        if result.returncode != 0:
            if os.path.exists(proof_file):
                os.unlink(proof_file)
            if os.path.exists(public_file):
                os.unlink(public_file)
            return None

        with open(public_file, 'r') as f:
            public_data = json.load(f)

        os.unlink(proof_file)
        os.unlink(public_file)

        return {
            'goalAchieved': int(public_data[0]),
            'verifiedWeekTimestamp': int(public_data[1]),
            'totalEquivalentMinutes': int(public_data[2]),
            'allSessionsConsistent': int(public_data[3])
        }
    except Exception:
        return None


def create_standard_input(week_timestamp, features_fn, gap_seconds=60):
    base_time = 1702000000
    timestamps = [
        create_session_timestamps(base_time + i * 10000, READINGS_PER_SESSION, gap_seconds)
        for i in range(N_SESSIONS)
    ]
    features = [features_fn() for _ in range(N_SESSIONS)]

    return {
        "weekTimestamp": week_timestamp,
        "timestamps": timestamps,
        "features": features
    }


class TestDecisionTreeClassification:

    def test_rest_classification(self):
        input_data = create_standard_input("1733097600", create_rest_features)

        witness_file, error = generate_witness(input_data)
        assert error is None, f"Witness generation failed: {error}"

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals is not None
        assert signals['allSessionsConsistent'] == 1
        assert signals['totalEquivalentMinutes'] == 0
        assert signals['goalAchieved'] == 0

    def test_moderate_classification(self):
        input_data = create_standard_input("1733097600", create_moderate_features)

        witness_file, error = generate_witness(input_data)
        assert error is None, f"Witness generation failed: {error}"

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals is not None
        assert signals['allSessionsConsistent'] == 1
        assert signals['totalEquivalentMinutes'] > 0
        assert signals['totalEquivalentMinutes'] < 220

    def test_vigorous_classification(self):
        input_data = create_standard_input("1733097600", create_vigorous_features)

        witness_file, error = generate_witness(input_data)
        assert error is None, f"Witness generation failed: {error}"

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals is not None
        assert signals['allSessionsConsistent'] == 1
        assert signals['totalEquivalentMinutes'] == 220
        assert signals['goalAchieved'] == 1

    def test_mixed_classification(self):
        base_time = 1702000000
        timestamps = [
            create_session_timestamps(base_time + i * 10000, READINGS_PER_SESSION, 60)
            for i in range(N_SESSIONS)
        ]
        features = []
        for i in range(N_SESSIONS):
            if i < 5:
                features.append(create_vigorous_features())
            else:
                features.append(create_moderate_features())

        input_data = {
            "weekTimestamp": "1733097600",
            "timestamps": timestamps,
            "features": features
        }

        witness_file, error = generate_witness(input_data)
        assert error is None, f"Witness generation failed: {error}"

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals is not None
        assert signals['allSessionsConsistent'] == 1
        assert signals['goalAchieved'] == 1


class TestTimestampConsistency:

    def test_consistent_timestamps_pass(self):
        input_data = create_standard_input("1733097600", create_vigorous_features, gap_seconds=60)

        witness_file, error = generate_witness(input_data)
        assert error is None, f"Witness generation failed: {error}"

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals['allSessionsConsistent'] == 1

    def test_inconsistent_timestamps_fail(self):
        base_time = 1702000000
        timestamps = []
        for i in range(N_SESSIONS):
            session = create_session_timestamps(base_time + i * 10000, READINGS_PER_SESSION, 30)
            session[-1] = str(int(session[-2]) + 100)
            timestamps.append(session)

        features = [create_vigorous_features() for _ in range(N_SESSIONS)]

        input_data = {
            "weekTimestamp": "1733097600",
            "timestamps": timestamps,
            "features": features
        }

        witness_file, error = generate_witness(input_data)
        assert error is None, f"Witness generation failed: {error}"

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals['allSessionsConsistent'] == 0
        assert signals['goalAchieved'] == 0

    def test_non_increasing_timestamps_fail(self):
        base_time = 1702000000
        timestamps = []
        for i in range(N_SESSIONS):
            session = create_session_timestamps(base_time + i * 10000, READINGS_PER_SESSION, 30)
            session[5], session[6] = session[6], session[5]
            timestamps.append(session)

        features = [create_vigorous_features() for _ in range(N_SESSIONS)]

        input_data = {
            "weekTimestamp": "1733097600",
            "timestamps": timestamps,
            "features": features
        }

        witness_file, error = generate_witness(input_data)
        assert error is None, f"Witness generation failed: {error}"

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals['allSessionsConsistent'] == 0


class TestDurationCalculation:

    def test_11_minute_session_duration(self):
        input_data = create_standard_input("1733097600", create_vigorous_features, gap_seconds=60)

        witness_file, error = generate_witness(input_data)
        assert error is None

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals['totalEquivalentMinutes'] == 220

    def test_zero_duration_session(self):
        input_data = create_standard_input("1733097600", create_vigorous_features, gap_seconds=5)

        witness_file, error = generate_witness(input_data)
        assert error is None

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals['totalEquivalentMinutes'] == 0


class TestWHOGoalAchievement:

    def test_goal_achieved_with_vigorous(self):
        input_data = create_standard_input("1733097600", create_vigorous_features)

        witness_file, error = generate_witness(input_data)
        assert error is None

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals['totalEquivalentMinutes'] >= WHO_GOAL_EQUIVALENT_MINUTES
        assert signals['goalAchieved'] == 1

    def test_goal_not_achieved_with_rest(self):
        input_data = create_standard_input("1733097600", create_rest_features)

        witness_file, error = generate_witness(input_data)
        assert error is None

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals['totalEquivalentMinutes'] == 0
        assert signals['goalAchieved'] == 0

    def test_goal_not_achieved_inconsistent_timestamps(self):
        base_time = 1702000000
        timestamps = []
        for i in range(N_SESSIONS):
            session = create_session_timestamps(base_time + i * 10000, READINGS_PER_SESSION, 60)
            session[-1] = str(int(session[-2]) + 100)
            timestamps.append(session)

        features = [create_vigorous_features() for _ in range(N_SESSIONS)]

        input_data = {
            "weekTimestamp": "1733097600",
            "timestamps": timestamps,
            "features": features
        }

        witness_file, error = generate_witness(input_data)
        assert error is None

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals['allSessionsConsistent'] == 0
        assert signals['goalAchieved'] == 0


class TestWeekTimestampAntiReplay:

    def test_week_timestamp_preserved(self):
        week_ts = "1733097600"
        input_data = create_standard_input(week_ts, create_vigorous_features)

        witness_file, error = generate_witness(input_data)
        assert error is None

        signals = get_public_signals(witness_file)
        os.unlink(witness_file)

        assert signals['verifiedWeekTimestamp'] == int(week_ts)

    def test_different_week_timestamps_produce_different_outputs(self):
        week_ts_1 = "1733097600"
        week_ts_2 = "1733702400"

        input_1 = create_standard_input(week_ts_1, create_vigorous_features)
        input_2 = create_standard_input(week_ts_2, create_vigorous_features)

        witness_1, error = generate_witness(input_1)
        assert error is None
        signals_1 = get_public_signals(witness_1)
        os.unlink(witness_1)

        witness_2, error = generate_witness(input_2)
        assert error is None
        signals_2 = get_public_signals(witness_2)
        os.unlink(witness_2)

        assert signals_1['verifiedWeekTimestamp'] == int(week_ts_1)
        assert signals_2['verifiedWeekTimestamp'] == int(week_ts_2)
        assert signals_1['verifiedWeekTimestamp'] != signals_2['verifiedWeekTimestamp']


class TestDecisionTreeThresholds:

    def test_wrist_accel_std_threshold_rest_boundary(self):
        base_time = 1702000000
        timestamps = [
            create_session_timestamps(base_time + i * 10000, READINGS_PER_SESSION, 60)
            for i in range(N_SESSIONS)
        ]

        features_below = ["980", "600", "1300", "120", "145", "1100", "30", "160"]
        features_above = ["980", "600", "1300", "120", "145", "1100", "40", "160"]

        input_below = {
            "weekTimestamp": "1733097600",
            "timestamps": timestamps,
            "features": [features_below for _ in range(N_SESSIONS)]
        }
        input_above = {
            "weekTimestamp": "1733097600",
            "timestamps": timestamps,
            "features": [features_above for _ in range(N_SESSIONS)]
        }

        witness_below, _ = generate_witness(input_below)
        signals_below = get_public_signals(witness_below)
        os.unlink(witness_below)

        witness_above, _ = generate_witness(input_above)
        signals_above = get_public_signals(witness_above)
        os.unlink(witness_above)

        assert signals_below['totalEquivalentMinutes'] == 0
        assert signals_above['totalEquivalentMinutes'] > 0

    def test_chest_accel_std_threshold_vigorous(self):
        base_time = 1702000000
        timestamps = [
            create_session_timestamps(base_time + i * 10000, READINGS_PER_SESSION, 60)
            for i in range(N_SESSIONS)
        ]

        features_vigorous = ["980", "600", "1300", "120", "145", "1100", "65", "160"]

        input_data = {
            "weekTimestamp": "1733097600",
            "timestamps": timestamps,
            "features": [features_vigorous for _ in range(N_SESSIONS)]
        }

        witness, _ = generate_witness(input_data)
        signals = get_public_signals(witness)
        os.unlink(witness)

        assert signals['totalEquivalentMinutes'] == 220


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
