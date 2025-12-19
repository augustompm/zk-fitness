# ZK-Fitness

Privacy-preserving exercise verification using Zero-Knowledge Proofs.

## Overview

This system proves that a user met the WHO weekly exercise goal (150 equivalent minutes) without revealing health data. A Decision Tree classifier runs inside the cryptographic circuit.

## Structure

```
circuits/
  zkml_fitness_classifier.circom   # ZKP circuit with embedded ML
  input_example.json

src/ml/
  train_decision_tree.py           # Model training
  trained_tree.json                # Exported model

tests/
  unit/test_zkml_circuit.py
  integration/test_zkml_workflow.py
```

## References

- WHO Physical Activity Guidelines 2020
- ACSM Guidelines for Exercise Testing
- MHEALTH Dataset (Banos et al. 2014)
- Groth16 (EUROCRYPT 2016)

## License

MIT
