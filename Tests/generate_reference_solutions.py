"""
Generate Reference Solutions for Regression Testing

This script generates "gold standard" reference solutions that will be used
to detect if future code changes break physics correctness.

Reference cases:
1. Lima Flag (diffusion): N=120, 30 days
2. Cosine Bell (advection): N=120, 12 days (full orbit)

Run this script ONLY when you intentionally want to update reference solutions
(e.g., after fixing a bug or improving numerical accuracy).
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add solver paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'Solvers'))

from fv_cubesphere_diffusion import CubedSphereDiffusion
from fv_plr_cubesphere_adv import PLRCubeSphereAdvection


def generate_diffusion_reference(N=120, days=30.0, kappa=5e5):
    """Generate Lima Flag reference solution."""
    print("="*70)
    print("GENERATING DIFFUSION REFERENCE SOLUTION")
    print("="*70)
    print(f"Resolution: N={N}")
    print(f"Duration: {days} days")
    print(f"Diffusivity: κ={kappa:.2e} m²/s")
    
    # Config path (no sharding for deterministic results)
    config_path = Path(__file__).parent.parent / 'Config' / 'config_diffusion_no_sharding.yaml'
    
    # Create solver
    solver = CubedSphereDiffusion(N=N, kappa=kappa, config_file=str(config_path))
    
    # Initialize
    state = solver.initialize(pattern='quadrant', T_hot=600.0)
    
    # Time integration
    dt = 1800.0  # 30 min
    n_steps = int(days * 86400 / dt)
    
    print(f"\nRunning {n_steps} steps (dt={dt:.0f}s)...")
    
    for step in range(n_steps):
        state = solver.step(state, dt)
        
        if (step + 1) % 100 == 0:
            diag = solver.get_diagnostics(state)
            print(f"  Step {step+1:4d} / {n_steps}: " +
                  f"T_max={diag['T_max']:6.1f}K, " +
                  f"day={state.time/86400:.2f}")
    
    # Save reference solution
    output_file = Path(__file__).parent / 'validation' / 'diffusion_lima_flag_day30_N120.npz'
    
    np.savez_compressed(
        output_file,
        T=np.array(state.T),
        time=state.time,
        step=state.step,
        N=N,
        kappa=kappa,
        days=days,
        description="Lima Flag diffusion reference: N=120, 30 days, κ=5e5 m²/s"
    )
    
    print(f"\n✓ Reference saved: {output_file}")
    print(f"  Final T_max: {np.max(state.T):.2f} K")
    print(f"  Final time: {state.time/86400:.2f} days")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    return state


def generate_advection_reference(N=120, days=12.0):
    """Generate Cosine Bell reference solution."""
    print("\n" + "="*70)
    print("GENERATING ADVECTION REFERENCE SOLUTION")
    print("="*70)
    print(f"Resolution: N={N}")
    print(f"Duration: {days} days (1 full orbit)")
    
    # Config path (no sharding for deterministic results)
    config_path = Path(__file__).parent.parent / 'Config' / 'config_plr_advection_no_sharding.yaml'
    
    # Create solver
    solver = PLRCubeSphereAdvection(N=N, config_file=str(config_path))
    
    # Initialize (12-day rotation period)
    state = solver.initialize(test_case='cosine_bell', u0=None)
    
    # Compute timestep (CFL=0.5)
    import jax.numpy as jnp
    V_mag = jnp.sqrt(state.Vx**2 + state.Vy**2 + state.Vz**2)
    V_max = float(jnp.max(V_mag))
    dt = 0.5 * (solver.dx * 6.371e6) / V_max  # CFL=0.5
    n_steps = int(days * 86400 / dt)
    
    print(f"\nTime integration:")
    print(f"  dt={dt:.1f}s, n_steps={n_steps}")
    print(f"  V_max={V_max:.2f} m/s")
    
    # Run simulation
    diag0 = solver.get_diagnostics(state)
    mass_initial = diag0['mass']
    
    print(f"\nRunning {n_steps} steps...")
    
    for step in range(n_steps):
        state = solver.step(state, dt)
        
        if (step + 1) % 100 == 0:
            diag = solver.get_diagnostics(state)
            mass_err = abs(diag['mass'] - mass_initial) / mass_initial
            print(f"  Step {step+1:4d} / {n_steps}: " +
                  f"q_max={diag['q_max']:7.1f}, " +
                  f"mass_err={mass_err:.2e}, " +
                  f"day={state.time/86400:.2f}")
    
    # Save reference solution
    output_file = Path(__file__).parent / 'validation' / 'advection_cosine_bell_day12_N120.npz'
    
    np.savez_compressed(
        output_file,
        q=np.array(state.q),
        Vx=np.array(state.Vx),
        Vy=np.array(state.Vy),
        Vz=np.array(state.Vz),
        time=state.time,
        step=state.step,
        N=N,
        days=days,
        dt=dt,
        description="Cosine Bell advection reference: N=120, 12 days, CFL=0.5"
    )
    
    # Final diagnostics
    diag_final = solver.get_diagnostics(state)
    mass_err = abs(diag_final['mass'] - mass_initial) / mass_initial
    
    print(f"\n✓ Reference saved: {output_file}")
    print(f"  Final q_max: {diag_final['q_max']:.2f} (initial: {diag0['q_max']:.2f})")
    print(f"  Peak preservation: {diag_final['q_max']/diag0['q_max']:.4f}")
    print(f"  Mass conservation: {mass_err:.2e}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    return state


if __name__ == "__main__":
    print("\n" + "="*70)
    print("REFERENCE SOLUTION GENERATOR")
    print("="*70)
    print("This will create regression test reference data.")
    print("WARNING: Only run this when you want to UPDATE gold standards!")
    print("="*70)
    
    # Generate both reference solutions
    diff_state = generate_diffusion_reference(N=120, days=30.0)
    adv_state = generate_advection_reference(N=120, days=12.0)
    
    print("\n" + "="*70)
    print("✅ ALL REFERENCE SOLUTIONS GENERATED")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review output to confirm physics looks correct")
    print("  2. Run validation tests: pytest Tests/test_validation.py")
    print("  3. Commit these reference files to git (they're the gold standard!)")
    print("="*70 + "\n")

