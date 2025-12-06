"""
Main Execution Runner for JaxStream2 Framework

This is the primary entry point for running simulations.
It orchestrates solver initialization, time-stepping, I/O, and diagnostics.

Usage:
    python -m Framework.runner Config/my_config.yaml
    
Or from code:
    from Framework.runner import run_simulation
    state = run_simulation('Config/my_config.yaml')

Author: JaxStream2 Framework
"""

import sys
import yaml
import time
from pathlib import Path
from typing import Dict, Any, Optional

from Framework.solver_interface import NumericalSolver, validate_solver_interface
from Framework.io_manager import IOManager, validate_output_config


# ============================================================================
# SOLVER LOADER
# ============================================================================

def load_solver(config: Dict[str, Any]) -> NumericalSolver:
    """
    Dynamically load and instantiate solver from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Instantiated solver
    
    Raises:
        ValueError: If solver type unknown or invalid
    """
    solver_type = config['solver']['type']
    
    print(f"\n{'='*70}")
    print(f"LOADING SOLVER: {solver_type}")
    print(f"{'='*70}")
    
    # Import solver module
    sys.path.insert(0, str(Path(__file__).parent.parent / 'Solvers'))
    
    # Map solver types to (module_name, class_name, search_path)
    # search_path: 'Solvers' or 'Examples/subdir'
    SOLVER_REGISTRY = {
        'fv_cubesphere_diffusion': ('fv_cubesphere_diffusion', 'CubedSphereDiffusion', 'Solvers'),
        'fv_plr_cubesphere_adv': ('fv_plr_cubesphere_adv', 'PLRCubeSphereAdvection', 'Solvers'),
        'fv_torus_sw': ('solver', 'FVPeriodic2D', 'Examples/fv_periodic_2d'),  # Example solver
        # Add new solvers here:
        # 'fv_plr_cubesphere_swe': ('fv_plr_cubesphere_swe', 'ShallowWaterSolver', 'Solvers'),
    }
    
    if solver_type not in SOLVER_REGISTRY:
        available = list(SOLVER_REGISTRY.keys())
        raise ValueError(
            f"Unknown solver type: '{solver_type}'\n"
            f"Available solvers: {available}"
        )
    
    module_name, class_name, search_path = SOLVER_REGISTRY[solver_type]
    
    # Add appropriate directory to path
    if search_path.startswith('Examples'):
        sys.path.insert(0, str(Path(__file__).parent.parent / search_path))
    
    # Dynamic import
    try:
        solver_module = __import__(module_name)
        solver_class = getattr(solver_module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Failed to load solver '{solver_type}': {e}\n"
            f"Expected: {search_path}/{module_name}.py with class {class_name}"
        )
    
    # Instantiate solver (pass config for initialization)
    # Note: Different solvers have different __init__ signatures
    # We'll need to handle this gracefully
    solver = instantiate_solver(solver_class, config)
    
    # Validate interface
    validate_solver_interface(solver)
    print(f"  ✓ Solver interface validated")
    
    return solver


def instantiate_solver(solver_class, config: Dict[str, Any]):
    """
    Instantiate solver with appropriate arguments.
    
    Tries different argument patterns to support various solver signatures.
    
    Args:
        solver_class: Solver class to instantiate
        config: Full configuration dictionary
    
    Returns:
        Instantiated solver
    """
    # Try different initialization patterns
    
    # Pattern 1: Just config
    try:
        return solver_class(config=config)
    except TypeError:
        pass
    
    # Pattern 2: Specific parameters from config (legacy solvers)
    try:
        solver_config = config['solver']
        N = solver_config.get('N', 60)
        
        # Check if this is diffusion solver (needs kappa)
        if 'kappa' in solver_config:
            kappa = solver_config['kappa']
            config_file = config.get('_config_file_path', None)
            return solver_class(N=N, kappa=kappa, config_file=config_file)
        else:
            # Advection solver (just N and config file)
            config_file = config.get('_config_file_path', None)
            return solver_class(N=N, config_file=config_file)
    except Exception as e:
        raise ValueError(
            f"Failed to instantiate {solver_class.__name__}: {e}\n"
            f"Check solver's __init__ signature"
        )


# ============================================================================
# STATE INITIALIZATION
# ============================================================================

def initialize_solver_state(solver: NumericalSolver, config: Dict[str, Any]):
    """
    Initialize solver state from config.
    
    Handles different solver types with different initialization parameters.
    Reads initial condition settings from config['initial_condition'].
    
    Args:
        solver: Solver instance
        config: Full configuration dictionary
    
    Returns:
        Initial solver state
    """
    solver_type = config['solver']['type']
    ic_config = config.get('initial_condition', {})
    
    # Diffusion solver
    if solver_type == 'fv_cubesphere_diffusion':
        pattern = ic_config.get('pattern', 'quadrant')
        T_hot = ic_config.get('T_hot', 600.0)
        return solver.initialize(pattern=pattern, T_hot=T_hot)
    
    # Advection solver
    elif solver_type == 'fv_plr_cubesphere_adv':
        test_case = ic_config.get('test_case', 'cosine_bell')
        u0 = ic_config.get('u0', None)  # None = 12-day rotation
        return solver.initialize(test_case=test_case, u0=u0)
    
    # FVPeriodic2D (example solver)
    elif solver_type == 'fv_torus_sw':
        # This solver uses config dict directly
        return solver.initialize(config)
    
    # Generic fallback: try config dict, then no args
    else:
        try:
            return solver.initialize(config)
        except TypeError:
            return solver.initialize()


# ============================================================================
# MAIN SIMULATION DRIVER
# ============================================================================

def run_simulation(config_file: str, restart_from: Optional[int] = None) -> Any:
    """
    Main simulation driver.
    
    Args:
        config_file: Path to YAML configuration file
        restart_from: Specific checkpoint step to restart from (None = fresh start)
    
    Returns:
        Final solver state
    
    Workflow:
        1. Load configuration
        2. Load solver
        3. Setup I/O
        4. Initialize or restore state
        5. Time-stepping loop with diagnostics
        6. Finalize and return
    """
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Store config file path for solver initialization
    config['_config_file_path'] = config_file
    
    print(f"\n{'='*70}")
    print(f"JAXSTREAM2 FRAMEWORK")
    print(f"{'='*70}")
    print(f"Config: {config_file}")
    print(f"Solver: {config['solver']['type']}")
    
    # Load solver
    solver = load_solver(config)
    
    # Validate I/O config
    if 'io' in config:
        validate_output_config(solver, config)
        print(f"  ✓ I/O configuration validated")
    
    # Setup I/O
    io_manager = IOManager(solver, config)
    
    # Initialize or restore state
    if restart_from is not None or io_manager.checkpoint_exists():
        print(f"\n{'='*70}")
        print(f"RESTART MODE")
        print(f"{'='*70}")
        state = io_manager.restore_checkpoint(restart_from)
        if state is None:
            print("  ⚠ Restart failed, starting fresh")
            state = initialize_solver_state(solver, config)
        else:
            print(f"  ✓ Restarted from step {state.step}, time={state.time:.2f}s")
    else:
        state = initialize_solver_state(solver, config)
    
    # Initial diagnostics
    diag0 = solver.get_diagnostics(state)
    
    print(f"\n{'='*70}")
    print(f"INITIAL STATE")
    print(f"{'='*70}")
    for key, val in diag0.items():
        if isinstance(val, (int, float)):
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: {val}")
    
    # Time integration parameters
    dt = config['time_integration']['dt']
    num_steps = config['time_integration']['num_steps']
    start_step = state.step
    end_step = start_step + num_steps
    
    checkpoint_freq = config['io'].get('checkpoint_frequency', 500)
    diagnostic_freq = config['io'].get('diagnostic_frequency', 10)
    
    print(f"\n{'='*70}")
    print(f"TIME INTEGRATION")
    print(f"{'='*70}")
    print(f"  dt: {dt} s")
    print(f"  Steps: {start_step} → {end_step} ({num_steps} steps)")
    print(f"  Diagnostic frequency: every {diagnostic_freq} steps")
    print(f"  Checkpoint frequency: every {checkpoint_freq} steps")
    print(f"{'='*70}\n")
    
    # Time-stepping loop
    t_start = time.time()
    
    for step_count in range(num_steps):
        # Step forward
        state = solver.step(state, dt)
        
        # Write outputs (IOManager checks frequencies)
        io_manager.write_all_outputs(state)
        
        # Diagnostics
        if state.step % diagnostic_freq == 0:
            diag = solver.get_diagnostics(state)
            t_elapsed = time.time() - t_start
            steps_per_sec = (step_count + 1) / t_elapsed if t_elapsed > 0 else 0
            
            # Format diagnostic output (solver-specific)
            diag_str = format_diagnostics(diag, diag0)
            
            print(f"Step {state.step:5d} | t={state.time:8.1f}s | {diag_str} | "
                  f"{steps_per_sec:.2f} steps/s")
        
        # Checkpoint
        if state.step % checkpoint_freq == 0:
            io_manager.save_checkpoint(state)
    
    # Final diagnostics
    t_total = time.time() - t_start
    
    print(f"\n{'='*70}")
    print(f"SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {t_total:.2f} s")
    print(f"  Steps: {num_steps}")
    print(f"  Performance: {num_steps / t_total:.2f} steps/s")
    print(f"  Final simulation time: {state.time:.2f} s")
    
    # Save final checkpoint
    io_manager.save_checkpoint(state)
    
    # Finalize I/O
    io_manager.finalize()
    
    # Print output location
    print(f"\n📊 Output saved to:")
    print(f"  {io_manager.get_zarr_path()}")
    print(f"  {config['io']['checkpoint_dir']}/")
    print(f"{'='*70}\n")
    
    return state


def format_diagnostics(diag: Dict[str, Any], diag0: Dict[str, Any]) -> str:
    """
    Format diagnostics for logging.
    
    Tries to create a compact, informative string.
    Handles different solver types intelligently.
    
    Args:
        diag: Current diagnostics
        diag0: Initial diagnostics (for computing errors)
    
    Returns:
        Formatted string for logging
    """
    # Check what diagnostics are available
    if 'mass' in diag and 'energy' in diag:
        # Shallow water / conservation-based solver
        mass_err = (diag['mass'] - diag0['mass']) / diag0['mass']
        energy_err = (diag['energy'] - diag0['energy']) / diag0['energy']
        return (f"h=[{diag['h_min']:.2f}, {diag['h_max']:.2f}] | "
                f"ΔM={mass_err:+.2e} | ΔE={energy_err:+.2e}")
    
    elif 'heat_content' in diag:
        # Diffusion solver
        heat_err = (diag['heat_content'] - diag0['heat_content']) / diag0['heat_content']
        T_max = diag.get('T_max', 0)
        return f"T_max={T_max:6.1f}K | ΔH={heat_err:+.2e}"
    
    elif 'q_max' in diag:
        # Advection solver
        q_max = diag['q_max']
        q_min = diag.get('q_min', 0)
        mass_err = 0
        if 'mass' in diag and 'mass' in diag0:
            mass_err = (diag['mass'] - diag0['mass']) / diag0['mass']
        return f"q=[{q_min:.2f}, {q_max:.2f}] | ΔM={mass_err:+.2e}"
    
    else:
        # Generic fallback
        return str({k: v for k, v in diag.items() if k not in ['time', 'step']})


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='JaxStream2 Framework - Run numerical simulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('config', type=str,
                       help='Path to configuration YAML file')
    parser.add_argument('--restart', type=int, default=None,
                       help='Restart from specific checkpoint step (None = latest)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Validate config and solver, then exit')
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if args.validate_only:
        print("Running validation only...")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        solver = load_solver(config)
        validate_output_config(solver, config)
        print("✅ Configuration valid!")
        sys.exit(0)
    
    # Run simulation
    try:
        final_state = run_simulation(args.config, restart_from=args.restart)
        print("✅ Simulation completed successfully!")
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


# ============================================================================
# DOCUMENTATION
# ============================================================================

__doc__ += """

Command-Line Usage:
==================

# Run simulation
python -m Framework.runner Config/diffusion.yaml

# Restart from latest checkpoint
python -m Framework.runner Config/diffusion.yaml --restart

# Restart from specific step
python -m Framework.runner Config/diffusion.yaml --restart 1000

# Validate config without running
python -m Framework.runner Config/diffusion.yaml --validate-only


Python API Usage:
================

from Framework.runner import run_simulation

# Run simulation
final_state = run_simulation('Config/my_config.yaml')

# Access final state
print(f"Final time: {final_state.time}")
print(f"Final step: {final_state.step}")


Adding New Solvers:
==================

1. Create solver in Solvers/ (inherit from NumericalSolver)

2. Register in SOLVER_REGISTRY:
   SOLVER_REGISTRY = {
       'my_new_solver': ('module_name', 'ClassName'),
       ...
   }

3. Create config in Config/

4. Run:
   python -m Framework.runner Config/my_solver.yaml

That's it!
"""

