"""
I/O Manager for JaxStream2 Framework

Handles all file I/O operations:
- Zarr output for history/visualization
- Orbax checkpointing for restart capability

Design: Solver-driven I/O
- Solver declares what/when to save (OutputSpec)
- IOManager handles the mechanics of saving/loading
- Clean separation of concerns

Author: JaxStream2 Framework
"""

import zarr
import orbax.checkpoint as ocp
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from Framework.solver_interface import NumericalSolver, OutputSpec, SolverState


# ============================================================================
# I/O MANAGER
# ============================================================================

class IOManager:
    """
    Manages Zarr output and Orbax checkpointing for solvers.
    
    This class is solver-agnostic: it doesn't know about specific physics,
    just how to save/load arrays based on solver-provided specifications.
    
    Workflow:
        1. Solver provides OutputSpec (what variables, how often)
        2. IOManager creates Zarr datasets and checkpoint directories
        3. During time-stepping:
           - Framework calls write_output() at appropriate steps
           - Framework calls save_checkpoint() periodically
        4. For restart:
           - IOManager loads checkpoint data
           - Solver reconstructs state via state_from_checkpoint()
    
    Features:
        - Async checkpointing (non-blocking)
        - Compressed Zarr output (LZ4, fast)
        - Automatic dataset creation
        - Metadata tracking (time, step, config)
    """
    
    def __init__(self, solver: NumericalSolver, config: Dict[str, Any]):
        """
        Initialize IOManager.
        
        Args:
            solver: Solver instance (for output specs)
            config: Full configuration dictionary
        """
        self.solver = solver
        self.config = config
        
        # Setup directories
        output_dir = Path(config['io']['output_dir'])
        checkpoint_dir = Path(config['io']['checkpoint_dir'])
        
        output_dir.mkdir(exist_ok=True, parents=True)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Get output specifications from solver
        self.output_specs = solver.get_output_spec(config)
        
        # Setup Zarr for history output
        self.zarr_store = zarr.DirectoryStore(str(output_dir / 'output.zarr'))
        self.zarr_root = zarr.group(store=self.zarr_store, overwrite=True)
        self.datasets_created = {}  # Track which groups have datasets
        
        # Setup Orbax for checkpointing
        self.checkpoint_manager = ocp.CheckpointManager(
            str(checkpoint_dir.absolute()),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=config['io'].get('max_checkpoints', 3),
                create=True,
                enable_async_checkpointing=True  # Non-blocking saves
            )
        )
        
        print(f"📁 I/O Setup:")
        print(f"  Output: {output_dir}/output.zarr")
        print(f"  Checkpoints: {checkpoint_dir}/")
        print(f"  Output groups: {list(self.output_specs.keys())}")
        
        # Store metadata
        self._save_metadata(config)
    
    def _save_metadata(self, config: Dict[str, Any]):
        """Save run metadata to Zarr."""
        import json
        import datetime
        
        metadata = {
            'solver_type': config['solver']['type'],
            'config': config,
            'created': datetime.datetime.now().isoformat(),
            'jaxstream_version': '2.0'
        }
        
        self.zarr_root.attrs['metadata'] = json.dumps(metadata, indent=2)
    
    def create_datasets(self, output_group: str, sample_output: Dict[str, np.ndarray]):
        """
        Create Zarr datasets for an output group.
        
        Args:
            output_group: Name of output group (e.g., 'state', 'diagnostics')
            sample_output: Sample data to infer shapes/types
        """
        if output_group in self.datasets_created:
            return  # Already created
        
        group = self.zarr_root.create_group(output_group, overwrite=True)
        
        for var_name, data in sample_output.items():
            # Convert lists to numpy arrays
            if isinstance(data, list):
                data = np.array(data)
            
            if np.isscalar(data) or (isinstance(data, np.ndarray) and data.ndim == 0):
                # Scalar time series
                group.create_dataset(
                    var_name,
                    shape=(0,),
                    chunks=(1000,),
                    dtype=np.float64,
                    compressor=zarr.Blosc(cname='lz4', clevel=1)
                )
            else:
                # Array time series
                data = np.asarray(data)  # Ensure numpy array
                shape = (0,) + data.shape  # Add time dimension
                chunks = (1,) + data.shape  # One timestep per chunk
                
                group.create_dataset(
                    var_name,
                    shape=shape,
                    chunks=chunks,
                    dtype=data.dtype,
                    compressor=zarr.Blosc(cname='lz4', clevel=1)
                )
        
        self.datasets_created[output_group] = True
        print(f"  ✓ Created Zarr datasets for '{output_group}'")
    
    def write_output(self, state: SolverState, output_group: str):
        """
        Write output for a specific group.
        
        Args:
            state: Current solver state
            output_group: Name of output group to write
        """
        spec = self.output_specs.get(output_group)
        if spec is None or not spec.enabled:
            return
        
        if not spec.should_output(state.step):
            return
        
        # Get output data from solver
        output_data = self.solver.state_to_output(state, output_group)
        
        # Create datasets on first write
        if output_group not in self.datasets_created:
            self.create_datasets(output_group, output_data)
        
        # Append to each dataset
        group = self.zarr_root[output_group]
        
        for var_name, data in output_data.items():
            if var_name not in group:
                continue  # Skip metadata like 'time' if not in spec
            
            # Convert lists to numpy arrays
            if isinstance(data, list):
                data = np.array(data)
            
            ds = group[var_name]
            idx = ds.shape[0]
            
            if np.isscalar(data) or (isinstance(data, np.ndarray) and data.ndim == 0):
                # Scalar
                ds.resize(idx + 1)
                ds[idx] = data
            else:
                # Array
                data = np.asarray(data)
                ds.resize(idx + 1, *ds.shape[1:])
                ds[idx] = data
    
    def write_all_outputs(self, state: SolverState):
        """
        Write all enabled output groups (if due).
        
        Args:
            state: Current solver state
        """
        for output_group in self.output_specs.keys():
            self.write_output(state, output_group)
    
    def save_checkpoint(self, state: SolverState):
        """
        Save checkpoint for restart capability.
        
        Uses Orbax to save full solver state. This is bitwise-exact
        and handles JAX arrays, sharding, etc. automatically.
        
        Args:
            state: Current solver state
        """
        step = state.step
        
        # Orbax saves JAX pytrees directly (no conversion needed!)
        self.checkpoint_manager.save(
            step,
            args=ocp.args.StandardSave(state)
        )
        
        print(f"  💾 Checkpoint saved at step {step}")
    
    def restore_checkpoint(self, step: Optional[int] = None) -> Optional[SolverState]:
        """
        Restore state from checkpoint.
        
        Args:
            step: Specific step to restore (None = latest)
        
        Returns:
            Restored state, or None if no checkpoint exists
        """
        if step is None:
            step = self.checkpoint_manager.latest_step()
        
        if step is None:
            print("  ⚠ No checkpoint found")
            return None
        
        # Need a template state to restore into
        # Use the runner's helper function for proper initialization
        from Framework.runner import initialize_solver_state
        template_state = initialize_solver_state(self.solver, self.config)
        
        # Restore from checkpoint
        restored = self.checkpoint_manager.restore(
            step,
            args=ocp.args.StandardRestore(template_state)
        )
        
        print(f"  📂 Restored checkpoint from step {step}")
        return restored
    
    def checkpoint_exists(self) -> bool:
        """Check if any checkpoints exist."""
        return self.checkpoint_manager.latest_step() is not None
    
    def finalize(self):
        """
        Finalize I/O operations.
        
        Waits for async checkpoint writes to complete.
        Call this at the end of simulation.
        """
        print("\n  Finalizing I/O...")
        self.checkpoint_manager.wait_until_finished()
        print("  ✓ All I/O operations complete")
    
    def get_zarr_path(self) -> Path:
        """Get path to output Zarr store."""
        return Path(self.zarr_store.path) if hasattr(self.zarr_store, 'path') else None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_output_config(solver: NumericalSolver, config: Dict[str, Any]) -> bool:
    """
    Validate that config's I/O section is compatible with solver.
    
    Args:
        solver: Solver instance
        config: Configuration dictionary
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If config requests unavailable output groups
    """
    available = solver.get_available_outputs()
    requested = config.get('io', {}).get('output', {})
    
    for group_name in requested.keys():
        if group_name not in available:
            valid_groups = list(available.keys())
            raise ValueError(
                f"Config requests unknown output group '{group_name}'. "
                f"Available groups: {valid_groups}"
            )
    
    return True


def create_default_io_config(solver: NumericalSolver) -> Dict[str, Any]:
    """
    Create sensible default I/O config for a solver.
    
    Args:
        solver: Solver instance
    
    Returns:
        Default I/O configuration dictionary
    
    Example:
        default_config = create_default_io_config(my_solver)
        # User can override specific values
        default_config['io']['output']['state']['frequency'] = 100
    """
    available = solver.get_available_outputs()
    
    output_config = {}
    for group_name in available.keys():
        # Heuristic: state/geometry = infrequent, diagnostics = frequent
        if group_name in ['state', 'geometry']:
            freq = 100
        elif group_name == 'diagnostics':
            freq = 10
        else:
            freq = 50
        
        output_config[group_name] = {
            'frequency': freq,
            'enabled': True
        }
    
    return {
        'io': {
            'output_dir': './output',
            'checkpoint_dir': './checkpoints',
            'output': output_config,
            'checkpoint_frequency': 500,
            'max_checkpoints': 3
        }
    }


# ============================================================================
# DOCUMENTATION
# ============================================================================

__doc__ += """

Example Usage:
=============

# In Framework.runner:

solver = MySolver(config)
io_manager = IOManager(solver, config)

# Initialize
state = solver.initialize(config)

# Time loop
for step in range(num_steps):
    state = solver.step(state, dt)
    
    # Write outputs (IOManager checks frequencies internally)
    io_manager.write_all_outputs(state)
    
    # Checkpoint periodically
    if step % checkpoint_freq == 0:
        io_manager.save_checkpoint(state)

# Cleanup
io_manager.finalize()

# For restart:
if io_manager.checkpoint_exists():
    state = io_manager.restore_checkpoint()  # Latest checkpoint
    print(f"Restarting from step {state.step}")
"""

