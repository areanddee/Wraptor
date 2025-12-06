# 2D Periodic Shallow Water Example

**Reference implementation showing Framework integration**

## Overview

This example demonstrates a complete, working solver integrated with the JaxStream2 Framework:

- **Domain**: Periodic torus (doubly-periodic 2D domain)
- **Physics**: Shallow water equations with Coriolis
- **Method**: Rusanov (local Lax-Friedrichs) flux
- **Time integration**: RK2

## Test Case

**Geostrophic Jet with Perturbation**:
- Zonal jet in geostrophic balance
- Sinusoidal perturbation triggers instability
- Tests: Conservation, dynamics, Framework integration

## Running

```bash
# From JaxStream2/ directory
python -m Framework.runner Examples/fv_periodic_2d/config.yaml
```

## What This Example Shows

### 1. Complete Framework Integration
The solver (`solver.py`) implements all required methods:
- `initialize()` - Create initial condition
- `step()` - RK2 time integration
- `get_diagnostics()` - Conservation quantities
- `get_available_outputs()` - Define output groups
- `get_output_spec()` - Create output specs from config
- `state_to_output()` - Convert state to saveable format
- `state_from_checkpoint()` - Restore from checkpoint

### 2. Solver-Driven I/O
Config file controls **when** to save, solver controls **what**:
```yaml
io:
  output:
    state:          # Solver defines this group
      frequency: 12  # User controls frequency
      enabled: true
```

### 3. Restart Capability
```bash
# Run simulation
python -m Framework.runner Examples/fv_periodic_2d/config.yaml

# Restart from latest checkpoint
python -m Framework.runner Examples/fv_periodic_2d/config.yaml --restart
```

## Expected Output

```
======================================================================
JAXSTREAM2 FRAMEWORK
======================================================================
Config: Examples/fv_periodic_2d/config.yaml
Solver: fv_torus_sw

FVPeriodic2D Solver:
  Grid: 128 × 128
  Domain: 1000000.0 × 1000000.0 m
  Resolution: dx=7812.5 m, dy=7812.5 m
  Physics: g=9.81 m/s², f=0.0001 rad/s

======================================================================
INITIAL STATE
======================================================================
  mass: 1.28e+14
  energy: 6.15e+20
  ...

======================================================================
TIME INTEGRATION
======================================================================
  dt: 300 s
  Steps: 0 → 288 (288 steps)
  ...

Step    12 | t=   3600.0s | h=[989.74, 1009.18] | ΔM=+0.00e+00 | ΔE=+1.23e-10
Step    24 | t=   7200.0s | h=[989.73, 1009.19] | ΔM=+0.00e+00 | ΔE=+2.45e-10
...
```

## As a Template

Use this solver as a template for new solvers:

1. Copy `solver.py` → `Solvers/my_new_solver.py`
2. Modify physics/numerics
3. Register in `Framework/runner.py`:
   ```python
   SOLVER_REGISTRY = {
       'my_solver': ('my_new_solver', 'MySolver'),
       ...
   }
   ```
4. Create config in `Config/`
5. Run!

## Files

```
Examples/fv_periodic_2d/
├── solver.py          # Solver implementation
├── config.yaml        # Configuration file
└── README.md          # This file
```

## Physics Notes

**Geostrophic Balance**:
- Coriolis force balanced by pressure gradient
- Creates stable zonal jet
- Small perturbation → baroclinic instability

**Conservation**:
- Mass: Exact (machine precision)
- Energy: ~10⁻¹⁰ relative error (expected for RK2)

**Typical Run Time**:
- 1 day simulation (288 steps): ~2-5 seconds on CPU
- Mostly JIT compilation overhead (first step)
- Subsequent steps: ~10 ms/step

---

**This example serves as the reference for Framework integration!**

