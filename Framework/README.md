# JaxStream2 Framework

**Clean, modular infrastructure for numerical solvers on cubed-sphere grids**

---

## Overview

The JaxStream2 Framework provides a clean separation between:
- **Physics** (implemented in Solvers)
- **Infrastructure** (provided by Framework)

This architecture enables:
- ✅ Easy addition of new solvers
- ✅ Consistent I/O across all solvers
- ✅ Bomb-proof restart capability
- ✅ No code duplication
- ✅ Clear contracts via abstract base class

---

## Architecture

```
Framework/
├── solver_interface.py    # Abstract base class (NumericalSolver)
├── io_manager.py          # I/O utilities (Zarr + Orbax)
├── runner.py              # Main execution driver
├── verify_framework.py    # Verification script
└── README.md             # This file

Solvers/
├── fv_cubesphere_diffusion.py   # Production solver (inherits NumericalSolver)
├── fv_plr_cubesphere_adv.py     # Production solver (inherits NumericalSolver)
└── halo_exchange.py             # Shared utility

Config/
├── diffusion_default.yaml       # Configuration templates
└── advection_default.yaml

Examples/
├── lima_flag/                   # Complete working demo
└── cosine_bell/                 # Complete working demo
```

---

## Core Components

### 1. NumericalSolver (Abstract Base Class)

All solvers **must** inherit from `NumericalSolver` and implement:

```python
from Framework.solver_interface import NumericalSolver, OutputSpec

class MyAwesomeSolver(NumericalSolver):
    def initialize(self, config) -> SolverState:
        """Create initial state"""
    
    def step(self, state, dt) -> SolverState:
        """Advance one timestep"""
    
    def get_diagnostics(self, state) -> Dict[str, float]:
        """Compute monitoring quantities"""
    
    def get_available_outputs(self) -> Dict[str, List[str]]:
        """Define what variables can be saved"""
    
    def get_output_spec(self, config) -> Dict[str, OutputSpec]:
        """Create output specifications from config"""
    
    def state_to_output(self, state, output_group) -> Dict[str, np.ndarray]:
        """Convert state to saveable format"""
    
    def state_from_checkpoint(self, checkpoint_data) -> SolverState:
        """Restore state from checkpoint"""
```

**Why formal inheritance?**
- Enforces consistent interface
- Prevents solver drift
- Enables generic Framework operations
- Clear contract for new developers

### 2. IOManager (I/O Orchestration)

Handles file operations transparently:

```python
from Framework.io_manager import IOManager

io_manager = IOManager(solver, config)

# Automatic output based on solver's OutputSpec
io_manager.write_all_outputs(state)

# Bomb-proof checkpointing
io_manager.save_checkpoint(state)

# Bitwise-exact restart
restored_state = io_manager.restore_checkpoint()
```

**Features:**
- Zarr for history output (visualization)
- Orbax for checkpoints (bitwise reproducible)
- Async checkpointing (non-blocking)
- Automatic dataset creation
- Compression (LZ4, fast)

### 3. Runner (Main Driver)

Executes simulations with consistent workflow:

```python
from Framework.runner import run_simulation

# Run simulation
final_state = run_simulation('Config/my_config.yaml')

# Or from command line
# python -m Framework.runner Config/my_config.yaml
```

**Handles:**
- Config loading
- Solver instantiation
- I/O setup
- Time-stepping loop
- Diagnostics logging
- Checkpoint management
- Error handling

---

## Configuration Files

### Required Structure

```yaml
solver:
  type: fv_cubesphere_diffusion  # Registered solver name
  N: 120                         # Solver-specific parameters
  kappa: 5.0e5

parallelization:
  enable_sharding: true
  tiles_per_edge: 1
  device_type: cpu
  num_devices: 6

time_integration:
  dt: 1800        # Timestep [s]
  num_steps: 1440 # Total steps

io:
  output_dir: ./output
  checkpoint_dir: ./checkpoints
  output:
    state:           # Output group name (defined by solver)
      frequency: 48  # Save every 48 steps
      enabled: true
    diagnostics:
      frequency: 1   # Save every step
      enabled: true
  checkpoint_frequency: 500   # Save checkpoint every 500 steps
  diagnostic_frequency: 10    # Print diagnostics every 10 steps
  max_checkpoints: 3          # Keep last 3 checkpoints
```

### Output Control Philosophy

**Solver controls WHAT** (which variables):
```python
def get_available_outputs(self):
    return {
        'state': ['T', 'u', 'v'],         # Solver defines groups
        'diagnostics': ['mass', 'energy']
    }
```

**User controls WHEN** (frequency):
```yaml
io:
  output:
    state:
      frequency: 48  # User adjusts frequency
```

**Benefits:**
- User can't request invalid variables
- Solver maintains physics correctness
- Simple, non-error-prone configs

---

## Adding a New Solver

### Step 1: Create Solver Class

```python
# Solvers/my_new_solver.py

import jax
import jax.numpy as jnp
from flax import struct
from Framework.solver_interface import NumericalSolver, OutputSpec

@struct.dataclass
class MyState:
    q: jax.Array
    time: float
    step: int

class MyNewSolver(NumericalSolver):
    def __init__(self, config):
        # Or: def __init__(self, N, config_file=None):
        #     (Framework handles different signatures)
        self.N = config['solver']['N']
        # Setup grid, parameters, etc.
    
    def initialize(self, config):
        # Create initial condition
        q = jnp.zeros((6, self.N, self.N))
        return MyState(q=q, time=0.0, step=0)
    
    def step(self, state, dt):
        # Physics here
        q_new = state.q + dt * self.compute_rhs(state)
        return MyState(q=q_new, time=state.time+dt, step=state.step+1)
    
    def get_diagnostics(self, state):
        return {
            'mass': float(jnp.sum(state.q)),
            'q_max': float(jnp.max(state.q))
        }
    
    def get_available_outputs(self):
        return {
            'state': ['q'],
            'diagnostics': ['mass', 'q_max']
        }
    
    def get_output_spec(self, config):
        io_config = config['io']['output']
        available = self.get_available_outputs()
        return {
            'state': OutputSpec(
                variables=available['state'],
                frequency=io_config['state']['frequency'],
                enabled=io_config['state']['enabled']
            ),
            'diagnostics': OutputSpec(
                variables=available['diagnostics'],
                frequency=io_config['diagnostics']['frequency'],
                enabled=True
            )
        }
    
    def state_to_output(self, state, output_group):
        import numpy as np
        if output_group == 'state':
            return {'q': np.array(state.q), 'time': state.time, 'step': state.step}
        elif output_group == 'diagnostics':
            return self.get_diagnostics(state)
    
    def state_from_checkpoint(self, checkpoint_data):
        return MyState(
            q=jnp.array(checkpoint_data['q']),
            time=float(checkpoint_data['time']),
            step=int(checkpoint_data['step'])
        )
    
    def compute_rhs(self, state):
        # Your physics here
        pass
```

### Step 2: Register Solver

Edit `Framework/runner.py`:

```python
SOLVER_REGISTRY = {
    'my_new_solver': ('my_new_solver', 'MyNewSolver'),  # Add this line
    'fv_cubesphere_diffusion': ('fv_cubesphere_diffusion', 'CubedSphereDiffusion'),
    ...
}
```

### Step 3: Create Config

```yaml
# Config/my_new_solver.yaml
solver:
  type: my_new_solver
  N: 60

time_integration:
  dt: 100
  num_steps: 1000

io:
  output_dir: ./my_solver_output
  checkpoint_dir: ./my_solver_checkpoints
  output:
    state:
      frequency: 10
      enabled: true
    diagnostics:
      frequency: 1
      enabled: true
  checkpoint_frequency: 100
```

### Step 4: Run!

```bash
python -m Framework.runner Config/my_new_solver.yaml
```

That's it! Framework handles everything else.

---

## Testing

### Verify Framework

```bash
python Framework/verify_framework.py
```

Should show:
```
✅ ALL FRAMEWORK MODULES VERIFIED
```

### Validate Solver + Config

```bash
python -m Framework.runner Config/my_config.yaml --validate-only
```

### Run Regression Tests

```bash
python Tests/test_validation.py
```

---

## Best Practices

### DO:
✅ Inherit from `NumericalSolver` (formal, not loose)  
✅ Use `@struct.dataclass` for states (JAX compatible)  
✅ Keep physics in Solver, infrastructure in Framework  
✅ Let IOManager handle all file operations  
✅ Use Orbax for restart (bomb-proof, bitwise exact)  
✅ Test with validation suite after changes  

### DON'T:
❌ Mix demo code with Framework infrastructure  
❌ Duplicate I/O logic in solvers  
❌ Create custom checkpoint formats  
❌ Bypass Framework for "just this one case"  
❌ Update reference solutions to fix failing tests  

---

## Future Work

- [ ] Unit tests for Framework components
- [ ] Multi-solver coupling capability
- [ ] GPU/TPU device management
- [ ] Automatic performance profiling
- [ ] Live visualization hooks

---

## Design Philosophy

**Separation of Concerns:**
- Solver = Physics + Numerics
- Framework = Orchestration + I/O + Infrastructure

**Constraint Solves Problems:**
- Formal inheritance prevents drift
- OutputSpec prevents invalid I/O requests
- Abstract methods ensure completeness

**Flexibility Through Interfaces:**
- Solver controls what/when to save
- Framework controls how to save
- Clean extension points for new features

---

**Questions? See examples in `Examples/` or existing solvers in `Solvers/`**

