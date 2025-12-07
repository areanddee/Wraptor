"""
Ultra-short 5-step test to debug SWE solver.
Focus on conservation and RHS correctness.
"""

import sys
from pathlib import Path
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Framework.runner import load_solver


def main():
    config_path = Path(__file__).parent.parent.parent / "Config" / "swe_framework.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    config['_config_file_path'] = str(config_path)
    
    print("Loading solver and initializing...")
    solver = load_solver(config)
    state = solver.initialize(config)
    
    dt = 500  # seconds
    
    # Get initial diagnostics
    diag0 = solver.get_diagnostics(state)
    mass0 = diag0['mass']
    ke0 = diag0['kinetic']
    pe0 = diag0['geopotential']
    energy0 = ke0 + pe0
    
    print(f"\n{'='*70}")
    print(f"INITIAL STATE (t=0)")
    print(f"{'='*70}")
    print(f"  Mass:          {mass0:.6e}")
    print(f"  Kinetic:       {ke0:.6e}")
    print(f"  Geopotential:  {pe0:.6e}")
    print(f"  Total Energy:  {energy0:.6e}")
    
    # Take 5 steps
    print(f"\n{'='*70}")
    print(f"TAKING 5 STEPS (dt={dt}s)")
    print(f"{'='*70}")
    
    for step in range(1, 6):
        state = solver.step(state, dt)
        diag = solver.get_diagnostics(state)
        
        mass = diag['mass']
        ke = diag['kinetic']
        pe = diag['geopotential']
        energy = ke + pe
        
        mass_err = abs(mass - mass0) / mass0
        energy_err = abs(energy - energy0) / energy0
        
        print(f"\nStep {step} (t={state.time:.0f}s):")
        print(f"  Mass:          {mass:.6e}  (error: {mass_err:.6e})")
        print(f"  Kinetic:       {ke:.6e}")
        print(f"  Geopotential:  {pe:.6e}")
        print(f"  Total Energy:  {energy:.6e}  (error: {energy_err:.6e})")
        
        if mass_err > 0.01:
            print(f"  ⚠️  Mass error > 1%!")
        if energy_err > 0.01:
            print(f"  ⚠️  Energy error > 1%!")
    
    # Check field ranges
    h_arr = np.array(state.h)
    hu1_arr = np.array(state.hu1)
    hu2_arr = np.array(state.hu2)
    
    print(f"\n{'='*70}")
    print(f"FINAL FIELD RANGES")
    print(f"{'='*70}")
    print(f"  h:   min={h_arr.min():.2f} m, max={h_arr.max():.2f} m")
    print(f"  hu1: min={hu1_arr.min():.2f}, max={hu1_arr.max():.2f}")
    print(f"  hu2: min={hu2_arr.min():.2f}, max={hu2_arr.max():.2f}")
    
    if h_arr.min() < 0:
        print(f"\n❌ NEGATIVE DEPTH!")
    if np.any(np.isnan(h_arr)):
        print(f"\n❌ NaN DETECTED!")
    
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    if mass_err < 1e-6 and energy_err < 1e-3:
        print("✓ Excellent conservation")
    elif mass_err < 0.01 and energy_err < 0.1:
        print("⚠️  Acceptable but not great")
    else:
        print("❌ Poor conservation - solver needs work")


if __name__ == "__main__":
    main()

