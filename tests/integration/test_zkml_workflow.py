"""
test_zkml_workflow.py

Integration tests for the full ZKP workflow.
Tests compilation, witness generation, proof generation, and verification.

References:
    [14] Groth16 (EUROCRYPT 2016)
    [13] MHEALTH Dataset (Banos et al. 2014)
"""

import pytest
import subprocess
import json
import os
import time
import tempfile
from pathlib import Path

CIRCUIT_DIR = Path(__file__).parent.parent.parent / "circuits"
BUILD_DIR = CIRCUIT_DIR / "build"
WASM_FILE = BUILD_DIR / "zkml_fitness_classifier_js" / "zkml_fitness_classifier.wasm"
WITNESS_GEN = BUILD_DIR / "zkml_fitness_classifier_js" / "generate_witness.js"
ZKEY_FILE = BUILD_DIR / "zkml_fitness_classifier.zkey"
VKEY_FILE = BUILD_DIR / "verification_key.json"
R1CS_FILE = BUILD_DIR / "zkml_fitness_classifier.r1cs"


def create_valid_input():
    """Cria input válido que atinge a meta WHO."""
    base_time = 1702000000
    timestamps = []
    for i in range(10):
        session = [str(base_time + i * 10000 + j * 60) for j in range(12)]
        timestamps.append(session)

    features = [
        ["980", "600", "1300", "120", "145", "1100", "65", "160"]
        for _ in range(10)
    ]

    return {
        "weekTimestamp": "1733097600",
        "timestamps": timestamps,
        "features": features
    }


def create_invalid_timestamp_input():
    """Cria input com timestamps inconsistentes."""
    base_time = 1702000000
    timestamps = []
    for i in range(10):
        session = [str(base_time + i * 10000 + j * 60) for j in range(12)]
        session[-1] = str(int(session[-2]) + 100)
        timestamps.append(session)

    features = [
        ["980", "600", "1300", "120", "145", "1100", "65", "160"]
        for _ in range(10)
    ]

    return {
        "weekTimestamp": "1733097600",
        "timestamps": timestamps,
        "features": features
    }


class TestCircuitCompilation:
    """Testes de arquivos compilados do circuito."""

    def test_r1cs_exists(self):
        """Arquivo R1CS deve existir."""
        assert R1CS_FILE.exists(), f"R1CS não encontrado: {R1CS_FILE}"

    def test_wasm_exists(self):
        """Arquivo WASM deve existir."""
        assert WASM_FILE.exists(), f"WASM não encontrado: {WASM_FILE}"

    def test_zkey_exists(self):
        """Arquivo zkey deve existir."""
        assert ZKEY_FILE.exists(), f"zkey não encontrado: {ZKEY_FILE}"

    def test_verification_key_exists(self):
        """Arquivo de chave de verificação deve existir."""
        assert VKEY_FILE.exists(), f"Verification key não encontrado: {VKEY_FILE}"

    def test_verification_key_valid_json(self):
        """Chave de verificação deve ser JSON válido."""
        with open(VKEY_FILE, 'r') as f:
            vkey = json.load(f)

        assert 'protocol' in vkey
        assert vkey['protocol'] == 'groth16'
        assert 'vk_alpha_1' in vkey
        assert 'vk_beta_2' in vkey
        assert 'vk_gamma_2' in vkey
        assert 'vk_delta_2' in vkey


class TestWitnessGeneration:
    """Testes de geração de witness."""

    def test_witness_generation_valid_input(self):
        """Witness deve ser gerado com input válido."""
        input_data = create_valid_input()

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

            assert result.returncode == 0, f"Falha na geração: {result.stderr}"
            assert os.path.exists(witness_file)
            assert os.path.getsize(witness_file) > 0
        finally:
            os.unlink(input_file)
            if os.path.exists(witness_file):
                os.unlink(witness_file)

    def test_witness_generation_invalid_input_fails(self):
        """Witness deve falhar com input malformado."""
        invalid_input = {"weekTimestamp": "123"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_input, f)
            input_file = f.name

        witness_file = tempfile.NamedTemporaryFile(suffix='.wtns', delete=False).name

        try:
            result = subprocess.run(
                ['node', str(WITNESS_GEN), str(WASM_FILE), input_file, witness_file],
                capture_output=True,
                text=True,
                timeout=30
            )

            assert result.returncode != 0 or not os.path.exists(witness_file) or os.path.getsize(witness_file) == 0
        finally:
            os.unlink(input_file)
            if os.path.exists(witness_file):
                os.unlink(witness_file)


class TestProofGeneration:
    """Testes de geração de prova Groth16."""

    def test_proof_generation_succeeds(self):
        """Prova deve ser gerada com witness válido."""
        input_data = create_valid_input()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name

        witness_file = tempfile.NamedTemporaryFile(suffix='.wtns', delete=False).name
        proof_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
        public_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name

        try:
            subprocess.run(
                ['node', str(WITNESS_GEN), str(WASM_FILE), input_file, witness_file],
                capture_output=True,
                timeout=30
            )

            cmd = f'npx snarkjs groth16 prove "{ZKEY_FILE}" "{witness_file}" "{proof_file}" "{public_file}"'
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(BUILD_DIR),
                timeout=120,
                shell=True
            )

            assert result.returncode == 0, f"Falha na prova: {result.stderr}"
            assert os.path.exists(proof_file)
            assert os.path.exists(public_file)

            with open(proof_file, 'r') as f:
                proof = json.load(f)

            assert 'pi_a' in proof
            assert 'pi_b' in proof
            assert 'pi_c' in proof
            assert proof['protocol'] == 'groth16'

        finally:
            for f in [input_file, witness_file, proof_file, public_file]:
                if os.path.exists(f):
                    os.unlink(f)

    def test_proof_contains_correct_public_signals(self):
        """Prova deve conter sinais públicos corretos."""
        input_data = create_valid_input()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name

        witness_file = tempfile.NamedTemporaryFile(suffix='.wtns', delete=False).name
        proof_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
        public_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name

        try:
            subprocess.run(
                ['node', str(WITNESS_GEN), str(WASM_FILE), input_file, witness_file],
                capture_output=True,
                timeout=30
            )

            cmd = f'npx snarkjs groth16 prove "{ZKEY_FILE}" "{witness_file}" "{proof_file}" "{public_file}"'
            subprocess.run(cmd, shell=True, cwd=str(BUILD_DIR), timeout=120, capture_output=True)

            with open(public_file, 'r') as f:
                public = json.load(f)

            assert len(public) == 5
            assert public[0] == "1"
            assert public[1] == "1733097600"
            assert int(public[2]) >= 150
            assert public[3] == "1"

        finally:
            for f in [input_file, witness_file, proof_file, public_file]:
                if os.path.exists(f):
                    os.unlink(f)


class TestProofVerification:
    """Testes de verificação de prova."""

    def test_valid_proof_verifies(self):
        """Prova válida deve ser verificada com sucesso."""
        input_data = create_valid_input()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name

        witness_file = tempfile.NamedTemporaryFile(suffix='.wtns', delete=False).name
        proof_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
        public_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name

        try:
            subprocess.run(
                ['node', str(WITNESS_GEN), str(WASM_FILE), input_file, witness_file],
                capture_output=True,
                timeout=30
            )

            cmd_prove = f'npx snarkjs groth16 prove "{ZKEY_FILE}" "{witness_file}" "{proof_file}" "{public_file}"'
            subprocess.run(cmd_prove, shell=True, cwd=str(BUILD_DIR), timeout=120, capture_output=True)

            cmd_verify = f'npx snarkjs groth16 verify "{VKEY_FILE}" "{public_file}" "{proof_file}"'
            result = subprocess.run(
                cmd_verify,
                capture_output=True,
                text=True,
                cwd=str(BUILD_DIR),
                timeout=60,
                shell=True
            )

            assert result.returncode == 0
            assert 'OK' in result.stdout

        finally:
            for f in [input_file, witness_file, proof_file, public_file]:
                if os.path.exists(f):
                    os.unlink(f)

    def test_tampered_proof_fails_verification(self):
        """Prova adulterada deve falhar na verificação."""
        input_data = create_valid_input()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name

        witness_file = tempfile.NamedTemporaryFile(suffix='.wtns', delete=False).name
        proof_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
        public_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name

        try:
            subprocess.run(
                ['node', str(WITNESS_GEN), str(WASM_FILE), input_file, witness_file],
                capture_output=True,
                timeout=30
            )

            cmd_prove = f'npx snarkjs groth16 prove "{ZKEY_FILE}" "{witness_file}" "{proof_file}" "{public_file}"'
            subprocess.run(cmd_prove, shell=True, cwd=str(BUILD_DIR), timeout=120, capture_output=True)

            with open(proof_file, 'r') as f:
                proof = json.load(f)

            proof['pi_a'][0] = "123456789"

            with open(proof_file, 'w') as f:
                json.dump(proof, f)

            cmd_verify = f'npx snarkjs groth16 verify "{VKEY_FILE}" "{public_file}" "{proof_file}"'
            result = subprocess.run(
                cmd_verify,
                capture_output=True,
                text=True,
                cwd=str(BUILD_DIR),
                timeout=60,
                shell=True
            )

            assert 'INVALID' in result.stdout or result.returncode != 0

        finally:
            for f in [input_file, witness_file, proof_file, public_file]:
                if os.path.exists(f):
                    os.unlink(f)


class TestEndToEndWorkflow:
    """Testes do fluxo completo end-to-end."""

    def test_complete_workflow_goal_achieved(self):
        """Fluxo completo com meta atingida."""
        input_data = create_valid_input()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name

        witness_file = tempfile.NamedTemporaryFile(suffix='.wtns', delete=False).name
        proof_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
        public_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name

        try:
            start = time.time()

            witness_start = time.time()
            subprocess.run(
                ['node', str(WITNESS_GEN), str(WASM_FILE), input_file, witness_file],
                capture_output=True,
                timeout=30
            )
            witness_time = (time.time() - witness_start) * 1000

            prove_start = time.time()
            cmd_prove = f'npx snarkjs groth16 prove "{ZKEY_FILE}" "{witness_file}" "{proof_file}" "{public_file}"'
            subprocess.run(cmd_prove, shell=True, cwd=str(BUILD_DIR), timeout=120, capture_output=True)
            prove_time = (time.time() - prove_start) * 1000

            verify_start = time.time()
            cmd_verify = f'npx snarkjs groth16 verify "{VKEY_FILE}" "{public_file}" "{proof_file}"'
            result = subprocess.run(cmd_verify, shell=True, cwd=str(BUILD_DIR), timeout=60, capture_output=True, text=True)
            verify_time = (time.time() - verify_start) * 1000

            total_time = (time.time() - start) * 1000

            assert result.returncode == 0
            assert 'OK' in result.stdout

            with open(public_file, 'r') as f:
                public = json.load(f)

            assert public[0] == "1"

            print(f"\n--- Performance Metrics ---")
            print(f"Witness: {witness_time:.2f} ms")
            print(f"Prove: {prove_time:.2f} ms")
            print(f"Verify: {verify_time:.2f} ms")
            print(f"Total: {total_time:.2f} ms")

        finally:
            for f in [input_file, witness_file, proof_file, public_file]:
                if os.path.exists(f):
                    os.unlink(f)

    def test_complete_workflow_goal_not_achieved(self):
        """Fluxo completo com meta NÃO atingida (timestamps inválidos)."""
        input_data = create_invalid_timestamp_input()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name

        witness_file = tempfile.NamedTemporaryFile(suffix='.wtns', delete=False).name
        proof_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
        public_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name

        try:
            subprocess.run(
                ['node', str(WITNESS_GEN), str(WASM_FILE), input_file, witness_file],
                capture_output=True,
                timeout=30
            )

            cmd_prove = f'npx snarkjs groth16 prove "{ZKEY_FILE}" "{witness_file}" "{proof_file}" "{public_file}"'
            subprocess.run(cmd_prove, shell=True, cwd=str(BUILD_DIR), timeout=120, capture_output=True)

            cmd_verify = f'npx snarkjs groth16 verify "{VKEY_FILE}" "{public_file}" "{proof_file}"'
            result = subprocess.run(cmd_verify, shell=True, cwd=str(BUILD_DIR), timeout=60, capture_output=True, text=True)

            assert result.returncode == 0
            assert 'OK' in result.stdout

            with open(public_file, 'r') as f:
                public = json.load(f)

            assert public[0] == "0"
            assert public[3] == "0"

        finally:
            for f in [input_file, witness_file, proof_file, public_file]:
                if os.path.exists(f):
                    os.unlink(f)


class TestPerformanceMetrics:
    """Testes de métricas de performance."""

    def test_proof_time_under_60_seconds(self):
        """Tempo de geração de prova deve ser menor que 60 segundos."""
        input_data = create_valid_input()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name

        witness_file = tempfile.NamedTemporaryFile(suffix='.wtns', delete=False).name
        proof_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
        public_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name

        try:
            subprocess.run(
                ['node', str(WITNESS_GEN), str(WASM_FILE), input_file, witness_file],
                capture_output=True,
                timeout=30
            )

            start = time.time()
            cmd_prove = f'npx snarkjs groth16 prove "{ZKEY_FILE}" "{witness_file}" "{proof_file}" "{public_file}"'
            subprocess.run(cmd_prove, shell=True, cwd=str(BUILD_DIR), timeout=120, capture_output=True)
            prove_time = time.time() - start

            assert prove_time < 60, f"Tempo de prova muito alto: {prove_time:.2f}s"

        finally:
            for f in [input_file, witness_file, proof_file, public_file]:
                if os.path.exists(f):
                    os.unlink(f)

    def test_verification_time_under_5_seconds(self):
        """Tempo de verificação deve ser menor que 5 segundos."""
        input_data = create_valid_input()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name

        witness_file = tempfile.NamedTemporaryFile(suffix='.wtns', delete=False).name
        proof_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
        public_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name

        try:
            subprocess.run(
                ['node', str(WITNESS_GEN), str(WASM_FILE), input_file, witness_file],
                capture_output=True,
                timeout=30
            )

            cmd_prove = f'npx snarkjs groth16 prove "{ZKEY_FILE}" "{witness_file}" "{proof_file}" "{public_file}"'
            subprocess.run(cmd_prove, shell=True, cwd=str(BUILD_DIR), timeout=120, capture_output=True)

            start = time.time()
            cmd_verify = f'npx snarkjs groth16 verify "{VKEY_FILE}" "{public_file}" "{proof_file}"'
            subprocess.run(cmd_verify, shell=True, cwd=str(BUILD_DIR), timeout=60, capture_output=True)
            verify_time = time.time() - start

            assert verify_time < 5, f"Tempo de verificação muito alto: {verify_time:.2f}s"

        finally:
            for f in [input_file, witness_file, proof_file, public_file]:
                if os.path.exists(f):
                    os.unlink(f)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
