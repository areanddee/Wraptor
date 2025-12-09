"""
Convergence study for SWE Test Case 2.

Runs at N=30, 60, 120 with dt scaled by CFL to same final time.
Computes L1, L2, L∞ errors vs analytic solution.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Framework.runner import load_solver
from Solvers.geometry import CubedSphereGeometry
from Solvers.physics import PlanetParams
from Solvers.initial_conditions.swe_testcase2 import steady_geostrophic_flow


def compute_errors(state, geometry, planet, h0=8000.0, u0=40.0):
    """Compute L1, L2, Linf errors vs analytic solution."""
    # Reference solution
    h_ref, u_lon_ref, u_lat_ref = steady_geostrophic_flow(geometry, planet, h0=h0, u0=u0)
    
    # Computed solution
    h = np.array(state.h)
    hu1 = np.array(state.hu1)
    hu2 = np.array(state.hu2)
    u1 = hu1 / h
    u2 = hu2 / h
    
    # Errors (assume u1 ≈ u_lon, u2 ≈ u_lat for now - not exact on cubed sphere!)
    dh = h - h_ref
    du1 = u1 - u_lon_ref
    du2 = u2 - u_lat_ref
    
    # Norms
    sqrtG = geometry.sqrtG
    total_area = np.sum(sqrtG)
    
    # L1 (area-weighted mean absolute error)
    L1_h = np.sum(np.abs(dh) * sqrtG) / total_area
    L1_u1 = np.sum(np.abs(du1) * sqrtG) / total_area
    L1_u2 = np.sum(np.abs(du2) * sqrtG) / total_area
    
    # L2 (area-weighted RMS error)
    L2_h = np.sqrt(np.sum(dh**2 * sqrtG) / total_area)
    L2_u1 = np.sqrt(np.sum(du1**2 * sqrtG) / total_area)
    L2_u2 = np.sqrt(np.sum(du2**2 * sqrtG) / total_area)
    
    # Linf (max absolute error)
    Linf_h = np.max(np.abs(dh))
    Linf_u1 = np.max(np.abs(du1))
    Linf_u2 = np.max(np.abs(du2))
    
    return {
        'L1': {'h': L1_h, 'u1': L1_u1, 'u2': L1_u2},
        'L2': {'h': L2_h, 'u1': L2_u1, 'u2': L2_u2},
        'Linf': {'h': Linf_h, 'u1': Linf_u1, 'u2': Linf_u2}
    }


def run_convergence_test(N, final_time_hours=1.0, CFL=0.4):
    """
    Run SWE test at given resolution to fixed final time.
    
    Args:
        N: grid resolution per face
        final_time_hours: integration time [hours]
        CFL: CFL number (dt will be scaled accordingly)
    
    Returns:
        errors: dict with L1, L2, Linf norms
        diagnostics: mass/energy conservation stats
    """
    # Load base config
    config_path = Path(__file__).parent.parent.parent / "Config" / "swe_framework.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Modify for this run
    config['solver']['N'] = N
    config['_config_file_path'] = str(config_path)
    
    # Compute timestep based on CFL
    # For cubed sphere: dx = π/(2N) radians
    # Physical spacing: Δx = R·dx
    # Wave speed: c ≈ √(gh) ≈ 268 m/s
    # dt_max = CFL · Δx / c
    
    R = float(config['physics']['R_sphere'])
    g = float(config['physics']['gravity'])
    h0 = float(config.get('physics', {}).get('h0', 8000.0))
    
    dx_rad = np.pi / (2.0 * N)
    dx_phys = R * dx_rad
    c = np.sqrt(g * h0)
    u_max = 40.0  # from initial condition
    
    dt = CFL * dx_phys / (c + u_max)
    final_time = final_time_hours * 3600  # to seconds
    num_steps = int(final_time / dt)
    
    config['time_integration']['dt'] = int(dt)
    config['time_integration']['num_steps'] = num_steps
    
    # Set up output directory
    output_dir = Path(__file__).parent / f"output_convergence_N{N}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    config['io']['output_dir'] = str(output_dir)
    config['io']['checkpoint_dir'] = str(output_dir / "checkpoints")
    config['io']['output']['state']['enabled'] = False
    config['io']['output']['diagnostics']['enabled'] = False  # Disable I/O for speed
    
    # Save config
    tmp_config = output_dir / "swe_config.yaml"
    with open(tmp_config, 'w') as f:
        yaml.safe_dump(config, f)
    
    print(f"\n{'='*70}")
    print(f"RUNNING N={N}")
    print(f"{'='*70}")
    print(f"  Resolution: {N}×{N} per face")
    print(f"  dx: {dx_rad:.6f} rad = {dx_phys/1000:.2f} km")
    print(f"  dt: {dt:.2f} s (CFL={CFL:.2f})")
    print(f"  Steps: {num_steps} (to t={final_time_hours:.1f} hr)")
    
    # Load solver and run
    solver = load_solver(config)
    state = solver.initialize(config)
    
    diag0 = solver.get_diagnostics(state)
    mass0 = diag0['mass']
    energy0 = diag0['kinetic'] + diag0['geopotential']
    
    # Time integration
    for step in range(num_steps):
        state = solver.step(state, dt)
        if (step + 1) % max(1, num_steps // 10) == 0:
            print(f"  Step {step+1}/{num_steps} ({100*(step+1)/num_steps:.0f}%)")
    
    # Final diagnostics
    diagf = solver.get_diagnostics(state)
    massf = diagf['mass']
    energyf = diagf['kinetic'] + diagf['geopotential']
    
    mass_err = abs(massf - mass0) / mass0
    energy_err = abs(energyf - energy0) / energy0
    
    print(f"  ✓ Complete")
    print(f"  Mass error: {mass_err:.6e}")
    print(f"  Energy error: {energy_err:.6e}")
    
    # Compute errors vs analytic
    geometry = CubedSphereGeometry.create(N)
    planet = PlanetParams.from_config(config)
    errors = compute_errors(state, geometry, planet)
    
    diagnostics = {
        'mass_error': mass_err,
        'energy_error': energy_err,
        'dt': dt,
        'num_steps': num_steps
    }
    
    # Cleanup output
    shutil.rmtree(output_dir)
    
    return errors, diagnostics


def main():
    resolutions = [30, 60, 120]
    final_time_hours = 1.0  # 1 hour integration
    CFL = 0.4  # Conservative CFL
    
    print(f"\n{'='*70}")
    print(f"SWE TEST CASE 2 CONVERGENCE STUDY")
    print(f"{'='*70}")
    print(f"Final time: {final_time_hours} hours")
    print(f"CFL: {CFL}")
    print(f"Resolutions: {resolutions}")
    
    results = {}
    for N in resolutions:
        errors, diag = run_convergence_test(N, final_time_hours, CFL)
        results[N] = {'errors': errors, 'diagnostics': diag}
    
    # Print results table
    print(f"\n{'='*70}")
    print(f"CONVERGENCE RESULTS")
    print(f"{'='*70}")
    print(f"\n{'N':<6} {'dx(km)':<10} {'dt(s)':<10} {'L2(h)':<12} {'L2(u1)':<12} {'L2(u2)':<12} {'Mass Err':<12}")
    print(f"{'-'*70}")
    
    for N in resolutions:
        res = results[N]
        dx_km = 6.371e6 * np.pi / (2 * N) / 1000
        dt = res['diagnostics']['dt']
        L2_h = res['errors']['L2']['h']
        L2_u1 = res['errors']['L2']['u1']
        L2_u2 = res['errors']['L2']['u2']
        mass_err = res['diagnostics']['mass_error']
        
        print(f"{N:<6} {dx_km:<10.2f} {dt:<10.2f} {L2_h:<12.2e} {L2_u1:<12.2e} {L2_u2:<12.2e} {mass_err:<12.2e}")
    
    # Compute convergence rates
    print(f"\n{'='*70}")
    print(f"CONVERGENCE RATES (log2(error_coarse / error_fine))")
    print(f"{'='*70}")
    
    for i in range(len(resolutions) - 1):
        N_coarse = resolutions[i]
        N_fine = resolutions[i + 1]
        
        L2_h_coarse = results[N_coarse]['errors']['L2']['h']
        L2_h_fine = results[N_fine]['errors']['L2']['h']
        
        L2_u1_coarse = results[N_coarse]['errors']['L2']['u1']
        L2_u1_fine = results[N_fine]['errors']['L2']['u1']
        
        rate_h = np.log2(L2_h_coarse / L2_h_fine) if L2_h_fine > 0 else np.nan
        rate_u1 = np.log2(L2_u1_coarse / L2_u1_fine) if L2_u1_fine > 0 else np.nan
        
        print(f"  N={N_coarse} → N={N_fine}:")
        print(f"    Rate(h):  {rate_h:.2f} (expect ≈2 for 2nd order)")
        print(f"    Rate(u1): {rate_u1:.2f}")


if __name__ == "__main__":
    main()

