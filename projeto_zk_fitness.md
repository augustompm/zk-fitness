# ZK-Fitness: Exercise Verification with Zero-Knowledge Proofs

## Summary

System that verifies if a user met the WHO weekly exercise goal (150 equivalent minutes) without revealing health data. A Decision Tree classifier runs inside the cryptographic circuit, preventing fraud in intensity classification.

---

## 1. Problem

| Approach | Issue |
|----------|-------|
| Share data | Privacy violation |
| Trust user | Allows fraud |
| External ML + simple ZKP | Can fake classification |
| **ML inside ZKP** | **Verifiable and private** |

---

## 2. Main Contribution

First ZKP circuit that embeds an exercise intensity classifier trained on real sensor data.

- Gap confirmed through zkML literature analysis [23]
- Existing frameworks (vCNN [16], ZKML [17]) focus on CNNs/DNNs
- None applied to fitness classification with wearable data

---

## 3. Architecture

```
Sensors (accel/gyro) --> [ZKP: Classify + Verify Goal] --> Proof (800 bytes)
                                    |
                        Decision Tree (97.19% accuracy)
```

### Components

```
circuits/
  zkml_fitness_classifier.circom   # ZKP circuit with embedded ML
  input_example.json

src/ml/
  train_decision_tree.py           # Model training
  trained_tree.json                # Exported model

tests/
  unit/                            # Unit tests
  integration/                     # Integration tests

experiments/
  benchmark_zkml_circuit.py        # Performance benchmark
```

---

## 4. Machine Learning

### Dataset

MHEALTH (Mobile Health) [13]:
- 10 subjects
- Sensors: accelerometer and gyroscope on chest, ankle, and wrist
- 12 activities mapped to 3 intensity levels [12]

### Features

| Feature | Description |
|---------|-------------|
| chest_accel_mean | Chest accelerometer magnitude mean |
| chest_accel_std | Chest accelerometer magnitude std |
| ankle_accel_mean | Ankle accelerometer magnitude mean |
| ankle_accel_std | Ankle accelerometer magnitude std |
| ankle_gyro_mean | Ankle gyroscope magnitude mean |
| wrist_accel_mean | Wrist accelerometer magnitude mean |
| wrist_accel_std | Wrist accelerometer magnitude std |
| wrist_gyro_mean | Wrist gyroscope magnitude mean |

### Model

- Algorithm: Decision Tree CART [21] (scikit-learn, Gini criterion)
- Max depth: 5, Leaves: 11
- Validation: 10-fold stratified cross-validation [19][20]
- CV accuracy: 97.11% (+/- 0.57%)
- Holdout test accuracy: 97.19%

### Intensity Classes

| Class | Value | MHEALTH Activities |
|-------|-------|-------------------|
| REST | 0 | Standing, sitting, lying |
| MODERATE | 1 | Walking, climbing stairs |
| VIGOROUS | 2 | Running, jumping, cycling |

---

## 5. Circom Circuit

### Structure

```
zkml_fitness_classifier.circom
├── Num2Bits              # Bit conversion
├── LessThan              # Comparator <
├── LessEqThan            # Comparator <=
├── GreaterThan           # Comparator >
├── GreaterEqThan         # Comparator >=
├── Mux2                  # 2:1 Multiplexer
├── IntegerDivision       # Integer division
├── DecisionTreeClassifier    # ML MODEL
├── TimestampConsistencyChecker  # Timestamp validation
├── SessionClassifier     # Process one session
└── ZkMLFitnessClassifier # Main template
```

### Circuit Metrics

| Metric | Value |
|--------|-------|
| Constraints | 9,767 |
| Private inputs | 200 |
| Public outputs | 4 |
| Sessions | 10 |
| Readings per session | 12 |

---

## 6. Proof System (Groth16)

### Trusted Setup

Groth16 [14] requires a trusted setup in two phases:

**Phase 1 - Powers of Tau (universal):**
- Ceremony with multiple participants
- Each adds random entropy
- If ONE is honest and destroys "toxic waste" -> secure
- We use `pot14_final.ptau` from Hermez/Polygon

**Phase 2 - Circuit-specific setup:**
```bash
snarkjs groth16 setup circuit.r1cs pot14.ptau circuit.zkey
snarkjs zkey contribute circuit.zkey circuit_final.zkey
snarkjs zkey export verificationkey circuit_final.zkey verification_key.json
```

### Key Sizes

| Artifact | Size | Location |
|----------|------|----------|
| Proving key (zkey) | 5.6 MB | Prover |
| Verification key | 3.6 KB | Verifier |
| Powers of Tau | 18.9 MB | Setup only |

### Proof Size

| Artifact | Size |
|----------|------|
| proof.json | ~800 bytes |
| public.json | ~50 bytes |
| **Total** | **< 1 KB** |

---

## 7. Scientific Parameters

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Weekly goal | >= 150 equivalent min | WHO [1] |
| Equivalence | 1 min vigorous = 2 min moderate | WHO [1] |
| Moderate zone | 50-70% HRmax | ACSM [12] |
| Vigorous zone | 70-85% HRmax | ACSM [12] |
| Max timestamp gap | 60 seconds | Design |
| Cross-validation | 10-fold stratified | HAR literature [19][20] |
| Split criterion | Gini index | CART [21] |

---

## 8. Limitations

- Sensor data assumed authentic (requires TEE for production)
- Simple model (Decision Tree vs Deep Learning)
- No trusted time source for timestamps

### Future Work

- TEE integration (TrustZone) for data authenticity
- More complex models (MLP, CNN) with frameworks like EZKL
- On-chain verification in smart contracts
- Mobile app implementation

---

## 9. Conclusion

ZK-Fitness demonstrates the feasibility of embedding Machine Learning inside ZKP circuits for private fitness goal verification:

- **Privacy:** health data never revealed
- **Verifiability:** impossible to fake classification
- **Efficiency:** 800 byte proof in 1.7 seconds
- **Scalability:** shared setup, lightweight verification

---

## References

[1] WHO, "WHO Guidelines on Physical Activity and Sedentary Behaviour," 2020.

[12] ACSM, "ACSM's Guidelines for Exercise Testing and Prescription," 11th ed., 2021.

[13] O. Banos et al., "mHealthDroid: A Novel Framework for Agile Development of Mobile Health Applications," IWAAL, 2014.

[14] J. Groth, "On the Size of Pairing-based Non-interactive Arguments," EUROCRYPT, 2016.

[16] S. Lee et al., "vCNN: Verifiable Convolutional Neural Network based on zk-SNARKs," IEEE TDSC, 2024.

[17] D. Kang et al., "ZKML: An Optimizing Compiler for ML Inference in Zero-Knowledge Proofs," EuroSys, 2024.

[19] R. Fan et al., "Human Activity Recognition Using Decision Tree Classifier," ICACCI, 2014.

[20] A. S. Sukor et al., "Activity Recognition Using Accelerometer Sensor and Machine Learning," ISCAIE, 2018.

[21] L. Breiman et al., "Classification and Regression Trees," Wadsworth, 1984.

[23] Y. Sun et al., "zkML: An Optimizing System for ML Inference in Zero-Knowledge Proofs," 2025.

---

## Repository

https://github.com/augustompm/zk-fitness
