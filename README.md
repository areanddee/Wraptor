# JaxStream2 - JAX-based Atmospheric Solver Framework

**Production-ready cubed-sphere atmospheric dynamics solvers with JAX sharding support.**

## Quick Start

```bash
# Run Framework with diffusion solver
python -m Framework.runner Config/diffusion_framework.yaml

# Run Framework with advection solver
python -m Framework.runner Config/advection_framework.yaml

# Run tests
python Tests/run_all_tests.py
```

---

## Directory Structure

```
JaxStream2/
├── Framework/              # Core infrastructure
│   ├── solver_interface.py # Abstract base class (7 methods)
│   ├── io_manager.py       # Zarr + Orbax I/O
│   ├── runner.py           # Main simulation driver
│   └── README.md           # Framework documentation
│
├── Solvers/                # Production solvers
│   ├── fv_cubesphere_diffusion.py  # Thermal diffusion (Lima Flag)
│   ├── fv_plr_cubesphere_adv.py    # PLR advection (Cosine Bell)
│   └── halo_exchange.py            # Optimized halo exchange
│
├── Config/                 # Configuration files
│   ├── diffusion_framework.yaml    # Full Framework config
│   └── advection_framework.yaml    # Full Framework config
│
├── Tests/                  # Validation suite
│   ├── run_all_tests.py    # Test dashboard
│   ├── test_solvers.py     # Pytest tests
│   └── validation/         # Reference solutions
│
├── Examples/               # Tutorial examples
│   ├── lima_flag_diffusion/    # Diffusion demo
│   └── cosine_bell_advection/  # Advection demo
│
└── Analysis/               # Visualization tools
    └── visualize_*.py
```

---

## Framework Architecture

### NumericalSolver Interface

All solvers inherit from `NumericalSolver` and implement 7 methods:

```python
class NumericalSolver(ABC):
    def initialize(config) -> SolverState      # Initial conditions
    def step(state, dt) -> SolverState         # Time stepping
    def get_diagnostics(state) -> Dict         # Monitoring
    def get_available_outputs() -> Dict        # Declare output variables
    def get_output_spec(config) -> Dict        # Configure output frequency
    def state_to_output(state, group) -> Dict  # Convert state to arrays
    def state_from_checkpoint(data) -> State   # Restart support
```

### I/O Strategy

- **Zarr**: History output (compressed, chunked)
- **Orbax**: Checkpoints (bitwise-exact restart)
- **Solver-driven**: Solver declares what to save, config controls when

---

## Running Simulations

### Using Framework Runner

```bash
# Basic run
python -m Framework.runner Config/diffusion_framework.yaml

# Restart from checkpoint
python -m Framework.runner Config/diffusion_framework.yaml --restart

# Validate config only
python -m Framework.runner Config/diffusion_framework.yaml --validate-only
```

### Using Examples

```bash
cd Examples/lima_flag_diffusion
python run.py --grid-size 60 --days 10

cd Examples/cosine_bell_advection
python run.py --grid-size 60 --days 12
```

---

## Testing

```bash
# Quick tests (~40 seconds)
python Tests/run_all_tests.py

# Full tests with regression (~10 minutes)
python Tests/run_all_tests.py --full

# Sharding sweep (1, 2, 3, 6 devices)
python Tests/run_all_tests.py --sharding sweep

# Pytest (if installed)
pytest Tests/test_solvers.py -v --quick-only
```

---

## Configuration

### Parallelization

```yaml
parallelization:
  enable_sharding: true     # Enable JAX Mesh sharding
  tiles_per_edge: 1         # Tiles per cube edge (1 = 6 total tiles)
  device_type: 'cpu'        # 'cpu' or 'gpu'
  num_devices: 6            # Must evenly divide num_tiles
```

### Valid Device Counts

| tiles_per_edge | num_tiles | Valid num_devices |
|----------------|-----------|-------------------|
| 1 | 6 | 1, 2, 3, 6 |
| 2 | 24 | 1, 2, 3, 4, 6, 8, 12, 24 |
| 3 | 54 | 1, 2, 3, 6, 9, 18, 27, 54 |

### I/O Configuration

```yaml
io:
  output_dir: ./output
  checkpoint_dir: ./checkpoints
  output:
    state:
      frequency: 12    # Save state every 12 steps
      enabled: true
    diagnostics:
      frequency: 1     # Save diagnostics every step
      enabled: true
  checkpoint_frequency: 100
```

---

## Adding New Solvers

1. Create solver file in `Solvers/`:
```python
from Framework.solver_interface import NumericalSolver, OutputSpec

class MySolver(NumericalSolver):
    def initialize(self, config): ...
    def step(self, state, dt): ...
    # ... implement all 7 methods
```

2. Register in `Framework/runner.py`:
```python
SOLVER_REGISTRY = {
    'my_solver': ('my_solver_module', 'MySolver', 'Solvers'),
    ...
}
```

3. Create config file in `Config/`

4. Run: `python -m Framework.runner Config/my_solver.yaml`

---

## Current Solvers

### fv_cubesphere_diffusion
- Thermal diffusion on cubed-sphere
- Forward Euler time integration
- "Lima Flag" test case
- Heat conservation to machine precision

### fv_plr_cubesphere_adv
- PLR (Piecewise Linear Reconstruction) advection
- RK3 time integration
- "Cosine Bell" test case (Williamson Test 1)
- ~95% peak preservation after 12-day orbit

---

## Performance

| Solver | Resolution | Device | Steps/sec |
|--------|------------|--------|-----------|
| Diffusion | 60×60×6 | CPU | ~10 |
| Advection | 60×60×6 | CPU | ~15 |

---

**Version:** 2.0  
**Date:** December 2024  
**Status:** Production-ready ✅
