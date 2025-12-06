# Cosine Bell Advection Example

Demonstrates PLR (Piecewise Linear Reconstruction) advection on a cubed-sphere.
This is **Williamson Test Case 1**: solid-body rotation of a cosine bell.

## The Test Case

A smooth cosine bell is initialized and transported around the sphere by a solid-body rotation velocity field. After one complete orbit (12 days), the bell should return to its starting position.

Key metrics:
- **Mass conservation**: Should be preserved (within numerical tolerance)
- **Peak preservation**: PLR with MC limiter preserves ~95% of peak after 12 days

## Quick Start

```bash
cd Examples/cosine_bell_advection
python run.py
```

## Options

```bash
python run.py --grid-size 120 --days 24 --cfl 0.5
```

| Option | Default | Description |
|--------|---------|-------------|
| `--grid-size`, `-N` | 60 | Grid resolution per face |
| `--days` | 12 | Simulation length (12 = 1 orbit) |
| `--cfl` | 0.5 | CFL number for timestep |
| `--output`, `-o` | output | Output directory |

## Expected Results

At N=120 after 12 days (1 orbit):
- Peak preserved: ~95.6%
- Mass error: ~0.4%

## Output

Saves `final_state.npz` containing:
- `q`: Tracer concentration (6, N, N)
- `time`: Simulation time [s]
- `step`: Step count
- `N`, `days`: Parameters

## Reference

Williamson, D.L., et al. (1992). "A standard test set for numerical approximations 
to the shallow water equations in spherical geometry." J. Comp. Phys.

