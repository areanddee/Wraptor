# Lima Flag Diffusion Example

Demonstrates thermal diffusion on a cubed-sphere using the `CubedSphereDiffusion` solver.

## The Test Case

The "Lima Flag" pattern places hot regions (600K) in a checkerboard pattern on Face 0 (north pole), with the rest of the sphere at background temperature (1K). Heat diffuses across face boundaries, demonstrating correct halo exchange.

## Quick Start

```bash
cd Examples/lima_flag_diffusion
python run.py
```

## Options

```bash
python run.py --grid-size 120 --days 30 --kappa 5e5
```

| Option | Default | Description |
|--------|---------|-------------|
| `--grid-size`, `-N` | 60 | Grid resolution per face |
| `--days` | 10 | Simulation length |
| `--kappa` | 5e5 | Diffusion coefficient [m²/s] |
| `--output`, `-o` | output | Output directory |

## Expected Results

- Heat conserved to ~1e-10 (machine precision)
- T_max decreases from 600K as heat spreads
- After 30 days at N=120: T_max ≈ 467K

## Output

Saves `final_state.npz` containing:
- `T`: Temperature field (6, N, N)
- `time`: Simulation time [s]
- `step`: Step count
- `N`, `days`, `kappa`: Parameters

