# SWE Solver Implementation Status

## Development Architecture

All development files are in `Dev/swe_dev/` until verified:
- `edge_extrapolation.py` - Putman & Lin 3rd-order edge algorithms
- `halo_exchange_2deep.py` - 2-deep halo exchange for edge extrapolation
- Test and diagnostic scripts

After verification, move to:
- `Solvers/physics/` - for halo exchange modules
- `Solvers/` - for solver modifications

## Current State

### What Works
1. **Continuity equation** with prescribed velocities (Test Case 1):
   - Mass conservation: ~4e-7 relative error after 50 steps
   - Matches advection solver accuracy

2. **Coriolis/vorticity term**:
   - Uses analytical absolute vorticity for solid body rotation:
     `η = 2(Ω + u₀/R) · sin(lat)` (Eq 94 from Williamson et al.)
   - Matches analytical to machine precision at t=0

3. **Bernoulli function**: Properly includes `B = gh + K` where `K = |V|²/2`

### Known Issues

1. **Pressure gradient accuracy at cube edges**:
   - Interior cells: 2nd-order accurate with FV-PLR
   - Edge cells: Only 1st-order (uses 1-deep ghost cell values)
   - Results in ~20% geostrophic imbalance in Test Case 2

2. **MC limiter kills slopes near smooth extrema**:
   - At equator, h has a local minimum/maximum
   - Limiter sets slopes to zero → incorrect gradients

## Putman & Lin (2007) 3rd-Order Edge Extrapolation

From "Finite-volume transport on various cubed-sphere grids", JCP 227:55-78:

### Key Equations

**Equation 47** - Shared edge value (requires 2 cells from each side):
```
q_e = [7(q₁ʳ + q₁ˡ) - (q₂ʳ + q₂ˡ)] / 12
```

**Equation 49** - Cubic reconstruction at edge:
```
q_{e+} = (3q₁ + 11q₂ - 2m₂) / 14
```

### Implementation Plan

1. **2-Deep Halo Exchange** (CREATED: `halo_exchange_2deep.py`):
   - Extends field to (6, N+4, N+4) with 2 ghost cells per edge
   - Exchanges q₁ and q₂ from neighboring faces

2. **Edge Extrapolation** (CREATED: `edge_extrapolation.py`):
   - `compute_shared_edge_value()` - Eq 47
   - `compute_edge_reconstruction()` - Eq 49
   - `compute_limited_slopes_with_edge_extrap()` - Enhanced slope function

3. **Integration** (TODO):
   - Create development SWE solver that uses edge extrapolation
   - Test with Test Case 2 geostrophic balance
   - Verify convergence rate improves

4. **Corner Handling** (FUTURE):
   - Putman & Lin note 120° coordinate intersections at 8 corners
   - May need special treatment beyond edge extrapolation

### Files in `Dev/swe_dev/`

| File | Purpose |
|------|---------|
| `edge_extrapolation.py` | Putman & Lin Eq 47, 49 algorithms |
| `halo_exchange_2deep.py` | 2-deep ghost cell exchange |
| `diagnostic_plots.py` | Vorticity and balance error plots |
| `test_swe_refactored.py` | Main test driver |
| `test_geostrophic_balance.py` | Balance diagnostics |
| `IMPLEMENTATION_STATUS.md` | This file |

## References

1. Putman, W.M. and Lin, S.-J. (2007). "Finite-volume transport on various 
   cubed-sphere grids." J. Comp. Phys. 227: 55-78.

2. Williamson, D.L., Drake, J.B., Hack, J.J., Jakob, R., and Swarztrauber, P.N. (1992).
   "A standard test set for numerical approximations to the shallow water equations 
   in spherical geometry." J. Comp. Phys. 102: 211-224.

## Test Commands

```bash
cd Dev/swe_dev

# Test 2-deep halo exchange
python halo_exchange_2deep.py

# Test Case 1 with prescribed velocities (continuity only)
python test_swe_refactored.py --test-case testcase1 --prescribed-velocity --steps 100

# Test Case 2 full SWE
python test_swe_refactored.py --test-case testcase2 --steps 50

# Geostrophic balance check
python test_geostrophic_balance.py

# Vorticity check
python test_vorticity.py

# Generate diagnostic plots
python diagnostic_plots.py
```

