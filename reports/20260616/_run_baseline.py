"""Run decoupled baseline for J_A = N/2 and generate remaining data."""

import importlib.util
import sys
from pathlib import Path

local_path = Path(__file__).resolve().parent / "general_4param_omega_drive.py"
spec = importlib.util.spec_from_file_location(
    "general_4param_omega_drive", str(local_path)
)
module = importlib.util.module_from_spec(spec)
sys.modules["general_4param_omega_drive"] = module
spec.loader.exec_module(module)

# Decoupled baseline for J_A=N/2 (full ancilla)
print("=== Decoupled Baseline (J_A = N/2) ===")
module.run_decoupled_baseline(force=True, ancilla_dim=4)

# Decoupled baseline for J_A=1/2 (already exists, but re-verify)
print("\n=== Decoupled Baseline (J_A = 1/2) ===")
module.run_decoupled_baseline(force=True, ancilla_dim=2)

print("\nDone!")
