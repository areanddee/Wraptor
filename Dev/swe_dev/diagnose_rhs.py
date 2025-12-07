"""
Diagnose the RHS of the SWE solver to check magnitudes and CFL.
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
    
    # Extract fields
    h = np.array(state.h)
    hu1 = np.array(state.hu1)
    hu2 = np.array(state.hu2)
    u1 = hu1 / h
    u2 = hu2 / h
    
    # Geometry
    dx = solver.geometry.dx  # radians
    dx_phys = dx * solver.planet.R_sphere  # meters
    
    # Wave speeds
    c_gravity = np.sqrt(solver.g * h.mean())  # shallow water wave speed
    u_max = max(np.abs(u1).max(), np.abs(u2).max())
    
    # CFL estimate
    dt_cfl = dx_phys / (c_gravity + u_max)
    
    print(f"\n{'='*70}")
    print(f"INITIAL STATE DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"Resolution: N={solver.N}, dx={dx:.6f} rad = {dx_phys/1000:.1f} km")
    print(f"\nDepth field:")
    print(f"  h: min={h.min():.2f} m, max={h.max():.2f} m, mean={h.mean():.2f} m")
    print(f"\nVelocity field:")
    print(f"  u1: min={u1.min():.4f} m/s, max={u1.max():.4f} m/s")
    print(f"  u2: min={u2.min():.4f} m/s, max={u2.max():.4f} m/s")
    print(f"  |u|_max: {u_max:.4f} m/s")
    print(f"\nWave speeds:")
    print(f"  c_gravity (√(gh)): {c_gravity:.2f} m/s")
    print(f"  Total max speed: {c_gravity + u_max:.2f} m/s")
    print(f"\nCFL analysis:")
    print(f"  dt_cfl_max (dx/(c+u)): {dt_cfl:.2f} s ({dt_cfl/60:.2f} min)")
    print(f"  Current dt: {config['time_integration']['dt']} s")
    print(f"  CFL number: {config['time_integration']['dt'] / dt_cfl:.4f}")
    
    if config['time_integration']['dt'] > dt_cfl:
        print(f"\n⚠️  WARNING: dt is {config['time_integration']['dt'] / dt_cfl:.2f}x too large!")
        print(f"  Recommend: dt ≤ {int(dt_cfl * 0.5)} s (CFL ≤ 0.5)")
    else:
        print(f"\n✓ dt is within CFL limit")
    
    # Now take a single step and check RHS magnitude
    print(f"\n{'='*70}")
    print(f"SINGLE STEP TEST (dt={config['time_integration']['dt']}s)")
    print(f"{'='*70}")
    
    state_new = solver.step(state, config['time_integration']['dt'])
    
    h_new = np.array(state_new.h)
    hu1_new = np.array(state_new.hu1)
    hu2_new = np.array(state_new.hu2)
    
    # Check for NaNs
    if np.any(np.isnan(h_new)):
        print("❌ NaN detected in h after 1 step!")
    if np.any(np.isnan(hu1_new)):
        print("❌ NaN detected in hu1 after 1 step!")
    if np.any(np.isnan(hu2_new)):
        print("❌ NaN detected in hu2 after 1 step!")
    
    # Mass change
    mass_init = float(np.sum(h))
    mass_new = float(np.sum(h_new))
    mass_error = abs(mass_new - mass_init) / mass_init
    
    print(f"\nMass conservation:")
    print(f"  Initial: {mass_init:.6e}")
    print(f"  After 1 step: {mass_new:.6e}")
    print(f"  Relative error: {mass_error:.6e}")
    
    if mass_error > 1e-10:
        print(f"  ⚠️  Mass NOT conserved! (expect machine precision for steady state)")
    
    # Field changes
    dh = h_new - h
    print(f"\nField changes after 1 step:")
    print(f"  Δh: min={dh.min():.6e}, max={dh.max():.6e}, rms={np.sqrt(np.mean(dh**2)):.6e}")
    print(f"  Δh/h: min={dh.min()/h.mean():.6e}, max={dh.max()/h.mean():.6e}")
    
    if abs(dh.max()) > h.mean() * 0.01:
        print(f"  ⚠️  Large change in h (>1% of mean)!")
    
    print(f"\n{'='*70}")
    print(f"CONCLUSION")
    print(f"{'='*70}")
    if mass_error > 1e-6 or abs(dh.max()) > h.mean() * 0.01:
        print("❌ Solver step function is BROKEN - mass not conserved and/or large spurious changes")
        print("   → Check flux-form discretization and metric terms")
    elif config['time_integration']['dt'] > dt_cfl:
        print("⚠️  CFL condition violated - reduce dt")
    else:
        print("✓ Solver appears stable for this timestep")


if __name__ == "__main__":
    main()

