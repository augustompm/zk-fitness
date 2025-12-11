"""
benchmark_zkml_circuit.py

Performance benchmark for the zkML fitness classification circuit.
Uses MHEALTH dataset for testing.

References:
    [13] MHEALTH Dataset (Banos et al. 2014)
    [14] Groth16 (EUROCRYPT 2016)
"""

import numpy as np
import json
import subprocess
import tempfile
import os
import time
from pathlib import Path
from datetime import datetime

MHEALTH_DIR = Path(__file__).parent.parent.parent / "datasets" / "MHEALTH_extracted" / "MHEALTHDATASET"

CIRCUIT_DIR = Path(__file__).parent.parent / "circuits"
BUILD_DIR = CIRCUIT_DIR / "build"
WASM_FILE = BUILD_DIR / "zkml_fitness_classifier_js" / "zkml_fitness_classifier.wasm"
WITNESS_GEN = BUILD_DIR / "zkml_fitness_classifier_js" / "generate_witness.js"
ZKEY_FILE = BUILD_DIR / "zkml_fitness_classifier.zkey"
VKEY_FILE = BUILD_DIR / "verification_key.json"

RESULTS_DIR = Path(__file__).parent.parent / "results"

CHEST_ACCEL_COLS = [0, 1, 2]
ANKLE_ACCEL_COLS = [5, 6, 7]
ANKLE_GYRO_COLS = [8, 9, 10]
WRIST_ACCEL_COLS = [14, 15, 16]
WRIST_GYRO_COLS = [17, 18, 19]
LABEL_COL = 23

WINDOW_SIZE = 50
SCALE_FACTOR = 100

ACTIVITY_TO_INTENSITY = {
    0: -1,
    1: 0, 2: 0, 3: 0,
    4: 1,
    5: 2,
    6: 1, 7: 1, 8: 1, 9: 1,
    10: 2, 11: 2, 12: 2,
}


def compute_magnitude(data):
    """Calcula magnitude do vetor 3D."""
    return np.sqrt(np.sum(data ** 2, axis=1))


def extract_features(window_data):
    """Extrai features de uma janela de dados no formato do circuito."""
    chest_accel = window_data[:, CHEST_ACCEL_COLS]
    ankle_accel = window_data[:, ANKLE_ACCEL_COLS]
    ankle_gyro = window_data[:, ANKLE_GYRO_COLS]
    wrist_accel = window_data[:, WRIST_ACCEL_COLS]
    wrist_gyro = window_data[:, WRIST_GYRO_COLS]

    chest_mag = compute_magnitude(chest_accel)
    ankle_accel_mag = compute_magnitude(ankle_accel)
    ankle_gyro_mag = compute_magnitude(ankle_gyro)
    wrist_accel_mag = compute_magnitude(wrist_accel)
    wrist_gyro_mag = compute_magnitude(wrist_gyro)

    features = [
        int(np.mean(chest_mag) * SCALE_FACTOR),
        int(np.std(chest_mag) * SCALE_FACTOR),
        int(np.mean(ankle_accel_mag) * SCALE_FACTOR),
        int(np.std(ankle_accel_mag) * SCALE_FACTOR),
        int(np.mean(ankle_gyro_mag) * SCALE_FACTOR),
        int(np.mean(wrist_accel_mag) * SCALE_FACTOR),
        int(np.std(wrist_accel_mag) * SCALE_FACTOR),
        int(np.mean(wrist_gyro_mag) * SCALE_FACTOR),
    ]

    return features


def load_subject_data(subject_id):
    """Carrega dados de um sujeito do MHEALTH."""
    file_path = MHEALTH_DIR / f"mHealth_subject{subject_id}.log"
    if not file_path.exists():
        return None
    return np.loadtxt(file_path)


def create_sessions_from_subject(data, num_sessions=10, readings_per_session=12):
    """Cria sessões de exercício a partir dos dados de um sujeito."""
    sessions_timestamps = []
    sessions_features = []

    exercise_indices = []
    labels = data[:, LABEL_COL].astype(int)

    for i in range(len(labels) - WINDOW_SIZE):
        label = labels[i]
        intensity = ACTIVITY_TO_INTENSITY.get(label, -1)
        if intensity > 0:
            exercise_indices.append(i)

    if len(exercise_indices) < num_sessions * readings_per_session:
        exercise_indices = list(range(0, len(data) - WINDOW_SIZE))

    step = max(1, len(exercise_indices) // (num_sessions * readings_per_session))

    base_timestamp = 1702000000
    session_gap = 3600

    for session_idx in range(num_sessions):
        timestamps = []
        features_list = []

        for reading_idx in range(readings_per_session):
            data_idx_position = session_idx * readings_per_session + reading_idx
            if data_idx_position * step < len(exercise_indices):
                data_idx = exercise_indices[data_idx_position * step]
            else:
                data_idx = exercise_indices[-1]

            window = data[data_idx:data_idx + WINDOW_SIZE]
            if len(window) < WINDOW_SIZE:
                window = data[-WINDOW_SIZE:]

            features = extract_features(window)
            features_list.append([str(f) for f in features])

            timestamp = base_timestamp + session_idx * session_gap + reading_idx * 60
            timestamps.append(str(timestamp))

        sessions_timestamps.append(timestamps)
        sessions_features.append(features_list[0])

    return sessions_timestamps, sessions_features


def run_zkp_workflow(input_data):
    """Executa workflow ZKP completo e retorna métricas de tempo."""
    metrics = {}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(input_data, f)
        input_file = f.name

    witness_file = tempfile.NamedTemporaryFile(suffix='.wtns', delete=False).name
    proof_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
    public_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name

    try:
        start = time.time()
        result = subprocess.run(
            ['node', str(WITNESS_GEN), str(WASM_FILE), input_file, witness_file],
            capture_output=True,
            timeout=30
        )
        metrics['witness_ms'] = (time.time() - start) * 1000

        if result.returncode != 0:
            return None, "Witness generation failed"

        start = time.time()
        cmd = f'npx snarkjs groth16 prove "{ZKEY_FILE}" "{witness_file}" "{proof_file}" "{public_file}"'
        result = subprocess.run(cmd, shell=True, cwd=str(BUILD_DIR), timeout=120, capture_output=True)
        metrics['prove_ms'] = (time.time() - start) * 1000

        if result.returncode != 0:
            return None, "Proof generation failed"

        start = time.time()
        cmd = f'npx snarkjs groth16 verify "{VKEY_FILE}" "{public_file}" "{proof_file}"'
        result = subprocess.run(cmd, shell=True, cwd=str(BUILD_DIR), timeout=60, capture_output=True, text=True)
        metrics['verify_ms'] = (time.time() - start) * 1000

        if result.returncode != 0 or 'OK' not in result.stdout:
            return None, "Verification failed"

        with open(public_file, 'r') as f:
            public = json.load(f)

        metrics['goal_achieved'] = int(public[0])
        metrics['equivalent_minutes'] = int(public[2])
        metrics['all_consistent'] = int(public[3])
        metrics['total_ms'] = metrics['witness_ms'] + metrics['prove_ms'] + metrics['verify_ms']

        return metrics, None

    finally:
        for f in [input_file, witness_file, proof_file, public_file]:
            if os.path.exists(f):
                os.unlink(f)


def run_benchmark(num_iterations=10):
    """Executa benchmark completo."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H%M%S")

    results_dir = RESULTS_DIR / date_str
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Benchmark zkML Fitness Classifier")
    print("=" * 60)
    print(f"Circuit: zkml_fitness_classifier.circom")
    print(f"Model: Decision Tree (97.19% accuracy)")
    print(f"Features: 8 (multi-sensor)")
    print(f"Sessions: 10 x 12 readings")
    print(f"Iterations: {num_iterations}")
    print("=" * 60)

    all_metrics = []
    successful = 0
    failed = 0

    for iteration in range(num_iterations):
        subject_id = (iteration % 10) + 1
        print(f"\n[{iteration + 1}/{num_iterations}] Subject {subject_id}...")

        data = load_subject_data(subject_id)
        if data is None:
            print(f"  Skipping - data not found")

            base_time = 1702000000
            timestamps = [[str(base_time + i * 10000 + j * 60) for j in range(12)] for i in range(10)]
            features = [["980", "600", "1300", "120", "145", "1100", "65", "160"] for _ in range(10)]
        else:
            timestamps, features = create_sessions_from_subject(data)

        input_data = {
            "weekTimestamp": str(1733097600 + iteration * 604800),
            "timestamps": timestamps,
            "features": features
        }

        metrics, error = run_zkp_workflow(input_data)

        if error:
            print(f"  Error: {error}")
            failed += 1
            continue

        successful += 1
        all_metrics.append(metrics)

        print(f"  Witness: {metrics['witness_ms']:.1f}ms")
        print(f"  Prove:   {metrics['prove_ms']:.1f}ms")
        print(f"  Verify:  {metrics['verify_ms']:.1f}ms")
        print(f"  Total:   {metrics['total_ms']:.1f}ms")
        print(f"  Goal:    {'YES' if metrics['goal_achieved'] else 'NO'} ({metrics['equivalent_minutes']} min)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_metrics:
        witness_times = [m['witness_ms'] for m in all_metrics]
        prove_times = [m['prove_ms'] for m in all_metrics]
        verify_times = [m['verify_ms'] for m in all_metrics]
        total_times = [m['total_ms'] for m in all_metrics]

        print(f"Successful: {successful}/{num_iterations} ({100*successful/num_iterations:.1f}%)")
        print(f"\nTiming (ms):")
        print(f"  Witness: {np.mean(witness_times):.1f} ± {np.std(witness_times):.1f}")
        print(f"  Prove:   {np.mean(prove_times):.1f} ± {np.std(prove_times):.1f}")
        print(f"  Verify:  {np.mean(verify_times):.1f} ± {np.std(verify_times):.1f}")
        print(f"  Total:   {np.mean(total_times):.1f} ± {np.std(total_times):.1f}")

        goals_achieved = sum(1 for m in all_metrics if m['goal_achieved'])
        print(f"\nGoals achieved: {goals_achieved}/{successful}")

        summary = {
            "experiment": "zkml_fitness_classifier_benchmark",
            "timestamp": datetime.now().isoformat(),
            "circuit": "zkml_fitness_classifier",
            "model": "DecisionTree",
            "model_accuracy": 0.9719,
            "features": 8,
            "iterations": num_iterations,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / num_iterations,
            "timing_ms": {
                "witness_mean": float(np.mean(witness_times)),
                "witness_std": float(np.std(witness_times)),
                "prove_mean": float(np.mean(prove_times)),
                "prove_std": float(np.std(prove_times)),
                "verify_mean": float(np.mean(verify_times)),
                "verify_std": float(np.std(verify_times)),
                "total_mean": float(np.mean(total_times)),
                "total_std": float(np.std(total_times)),
            },
            "goals_achieved": goals_achieved,
            "all_runs": all_metrics
        }

        output_file = results_dir / f"zkml_benchmark_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        return summary
    else:
        print("No successful runs!")
        return None


if __name__ == "__main__":
    import sys
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_benchmark(iterations)
