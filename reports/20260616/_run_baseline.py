"""Run decoupled baseline for J_A = N/2 and generate remaining data."""

import importlib

module = importlib.import_module("reports.20260616.general_4param_omega_drive")

# Decoupled baseline for J_A=N/2 (full ancilla)
print("=== Decoupled Baseline (J_A = N/2) ===")
module.run_decoupled_baseline(force=True, ancilla_dim=4)

# Decoupled baseline for J_A=1/2 (already exists, but re-verify)
print("\n=== Decoupled Baseline (J_A = 1/2) ===")
module.run_decoupled_baseline(force=True, ancilla_dim=2)

print("\nDone!")
