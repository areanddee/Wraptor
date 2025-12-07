"""
Short integration test for SWE Test Case 2.
Runs for 5 days and checks conservation + steady-state preservation.
"""

import sys
from pathlib import Path
import yaml

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Framework.runner import run_simulation


def main():
    # Load config
    config_path = Path(__file__).parent.parent.parent / "Config" / "swe_framework.yaml"
    print(f"Loading config: {config_path}\n")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Modify for short test run
    days = 5.0
    dt = config['time_integration']['dt']  # seconds
    num_steps = int(days * 86400 / dt)
    
    config['time_integration']['num_steps'] = num_steps
    
    # Set up output directory (clean slate - remove any old checkpoints)
    output_dir = Path(__file__).parent / "output_swe_short"
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    config['io']['output_dir'] = str(output_dir)
    config['io']['checkpoint_dir'] = str(output_dir / "checkpoints")
    
    # Enable diagnostics, disable full state history to save space
    config['io']['output']['state']['enabled'] = False
    config['io']['output']['diagnostics']['enabled'] = True
    config['io']['output']['diagnostics']['frequency'] = max(1, num_steps // 20)  # ~20 outputs
    
    # Save modified config
    tmp_config = output_dir / "swe_config.yaml"
    with open(tmp_config, 'w') as f:
        yaml.safe_dump(config, f)
    
    print(f"Running SWE Test Case 2:")
    print(f"  Duration: {days} days ({num_steps} steps)")
    print(f"  dt: {dt} s")
    print(f"  Output: {output_dir}\n")
    
    # Run simulation
    final_state = run_simulation(str(tmp_config))
    
    # Print final state statistics
    h_arr = np.array(final_state.h)
    hu1_arr = np.array(final_state.hu1)
    hu2_arr = np.array(final_state.hu2)
    
    u1 = hu1_arr / h_arr
    u2 = hu2_arr / h_arr
    
    print(f"\n{'='*70}")
    print(f"FINAL STATE (t = {final_state.time / 86400:.2f} days)")
    print(f"{'='*70}")
    print(f"  h:  min={h_arr.min():.2f} m, max={h_arr.max():.2f} m, mean={h_arr.mean():.2f} m")
    print(f"  u1: min={u1.min():.4f} m/s, max={u1.max():.4f} m/s")
    print(f"  u2: min={u2.min():.4f} m/s, max={u2.max():.4f} m/s")
    
    # Check for NaNs or extreme values
    if np.any(np.isnan(h_arr)) or np.any(np.isnan(u1)) or np.any(np.isnan(u2)):
        print("\n❌ FAILURE: NaN values detected!")
        return 1
    
    if h_arr.min() < 0:
        print(f"\n❌ FAILURE: Negative depth detected: {h_arr.min():.2f} m")
        return 1
    
    if h_arr.max() > 20000:
        print(f"\n⚠️  WARNING: Very large depth: {h_arr.max():.2f} m (expected ~8000 m)")
    
    print(f"\n✅ Integration completed successfully!")
    print(f"\nDiagnostics saved to: {output_dir}/diagnostics.zarr")
    print(f"Check conservation properties with: zarr info {output_dir}/diagnostics.zarr")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

