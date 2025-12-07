"""
Check individual tendencies in the SWE solver to isolate bugs.
Focus on u2 (meridional) which should have ZERO tendency.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Framework.runner import load_solver


def main():
    config_path = Path(__file__).parent.parent.parent / "Config" / "swe_framework.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    config['_config_file_path'] = str(config_path)
    
    print("Loading solver and initializing...\n")
    solver = load_solver(config)
    state = solver.initialize(config)
    
    # Extract fields
    h = np.array(state.h)
    hu1 = np.array(state.hu1)
    hu2 = np.array(state.hu2)
    u1 = hu1 / h
    u2 = hu2 / h
    
    print(f"{'='*70}")
    print(f"INITIAL STATE")
    print(f"{'='*70}")
    print(f"h:   min={h.min():.2f}, max={h.max():.2f}, mean={h.mean():.2f} m")
    print(f"u1:  min={u1.min():.4f}, max={u1.max():.4f} m/s")
    print(f"u2:  min={u2.min():.6f}, max={u2.max():.6f} m/s (should be ~0)")
    print(f"hu1: min={hu1.min():.2f}, max={hu1.max():.2f}")
    print(f"hu2: min={hu2.min():.6f}, max={hu2.max():.6f} (should be ~0)")
    
    # Check what the current step function computes
    print(f"\n{'='*70}")
    print(f"CURRENT STEP FUNCTION COMPUTATION")
    print(f"{'='*70}")
    
    # Replicate the buggy step function
    dt = 1.0  # Use dt=1s to see raw tendency magnitudes
    g = solver.g
    
    # What the current code does (WRONG):
    dh_wrong = -dt * (jnp.gradient(hu1, axis=1) + jnp.gradient(hu2, axis=2))
    dhu1_wrong = -dt * (jnp.gradient(hu1 * u1 + 0.5 * g * h * h, axis=1))
    dhu2_wrong = -dt * (jnp.gradient(hu2 * u2 + 0.5 * g * h * h, axis=2))
    
    dh_wrong = np.array(dh_wrong)
    dhu1_wrong = np.array(dhu1_wrong)
    dhu2_wrong = np.array(dhu2_wrong)
    
    print(f"\nUsing dt={dt}s (to see raw tendency magnitudes):")
    print(f"\nΔh (from jnp.gradient):")
    print(f"  min={dh_wrong.min():.6e}, max={dh_wrong.max():.6e}, rms={np.sqrt(np.mean(dh_wrong**2)):.6e}")
    
    print(f"\nΔhu1 (from jnp.gradient):")
    print(f"  min={dhu1_wrong.min():.6e}, max={dhu1_wrong.max():.6e}, rms={np.sqrt(np.mean(dhu1_wrong**2)):.6e}")
    
    print(f"\nΔhu2 (from jnp.gradient) - SHOULD BE ZERO:")
    print(f"  min={dhu2_wrong.min():.6e}, max={dhu2_wrong.max():.6e}, rms={np.sqrt(np.mean(dhu2_wrong**2)):.6e}")
    
    if abs(dhu2_wrong.max()) > 1e-10:
        print(f"  ❌ hu2 tendency is NOT zero! Something is wrong.")
    else:
        print(f"  ✓ hu2 tendency is zero (as expected)")
    
    # Check the individual terms
    print(f"\n{'='*70}")
    print(f"DEBUGGING INDIVIDUAL TERMS")
    print(f"{'='*70}")
    
    # Mass flux divergence components
    grad_hu1 = jnp.gradient(hu1, axis=1)
    grad_hu2 = jnp.gradient(hu2, axis=2)
    
    print(f"\n∂(hu1)/∂ξ1 (axis=1 gradient of hu1):")
    print(f"  min={np.array(grad_hu1).min():.6e}, max={np.array(grad_hu1).max():.6e}")
    
    print(f"\n∂(hu2)/∂ξ2 (axis=2 gradient of hu2):")
    print(f"  min={np.array(grad_hu2).min():.6e}, max={np.array(grad_hu2).max():.6e}")
    print(f"  NOTE: Since hu2=0 everywhere, this should be ~0")
    
    # Momentum flux + pressure gradient
    flux_term_1 = hu1 * u1 + 0.5 * g * h * h
    flux_term_2 = hu2 * u2 + 0.5 * g * h * h
    
    print(f"\nFlux term 1 (hu1*u1 + 0.5*g*h²):")
    print(f"  min={np.array(flux_term_1).min():.6e}, max={np.array(flux_term_1).max():.6e}")
    
    print(f"\nFlux term 2 (hu2*u2 + 0.5*g*h²):")
    print(f"  min={np.array(flux_term_2).min():.6e}, max={np.array(flux_term_2).max():.6e}")
    print(f"  NOTE: Since u2=0, this is just 0.5*g*h²")
    
    grad_flux2 = jnp.gradient(flux_term_2, axis=2)
    print(f"\n∂(flux_term_2)/∂ξ2:")
    print(f"  min={np.array(grad_flux2).min():.6e}, max={np.array(grad_flux2).max():.6e}")
    print(f"  This is ∂(0.5*g*h²)/∂ξ2 - the pressure gradient in the meridional direction")
    
    # Key issues
    print(f"\n{'='*70}")
    print(f"KEY ISSUES IDENTIFIED")
    print(f"{'='*70}")
    print(f"1. jnp.gradient() uses INDEX spacing (Δξ=1) not PHYSICAL spacing")
    print(f"   → Need to divide by dx (in radians) then multiply by R to get physical units")
    print(f"   → dx = {solver.geometry.dx:.6f} rad, R = {solver.planet.R_sphere:.0f} m")
    print(f"   → Scale factor: 1/(R*dx) = {1.0/(solver.planet.R_sphere * solver.geometry.dx):.6e}")
    
    print(f"\n2. No metric terms (√G) for flux-form discretization")
    print(f"   → Need ∂(F√G)/∂ξ / √G for proper conservative form")
    
    print(f"\n3. No halo exchange - gradients at face boundaries are WRONG")
    
    print(f"\n4. Pressure gradient in meridional direction is NON-ZERO")
    print(f"   → This will drive spurious meridional flow even though u2=0 initially")
    print(f"   → Need ∂(0.5*g*h²)/∂ξ2 = g*h*∂h/∂ξ2, which is NOT zero on the sphere!")


if __name__ == "__main__":
    main()

