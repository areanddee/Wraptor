# FV3 SWE Solver Implementation Plan: One-Sided Boundary Derivatives

**Date:** December 28, 2024  
**Status:** Root Cause Identified, Fix Verified, Ready for Integration

---

## Executive Summary

The SWE solver's gradient computation fails to converge at cube-sphere panel boundaries because it uses **centered differences with ghost cell values** at edges. The fix, per Ullrich et al. (2010), is to use **one-sided derivative stencils** at boundaries that require only interior cell data.

**Verified Results:**

| Method | N=15 Max Err | N=60 Max Err | Convergence Rate |
|--------|--------------|--------------|------------------|
| Baseline (MC + centered) | 19.1% | 23.5% | 0.44 |
| One-sided boundary | 1.4% | 0.09% | **2.44** |

---

## Root Cause Analysis

### The Problem

The current solver uses MC-limited centered differences for gradient reconstruction:

```python
slope = (f[i+1] - f[i-1]) / (2*dx)  # Requires ghost cell at boundary
```

At cube-sphere panel boundaries, the ghost cell `f[i-1]` or `f[i+1]` comes from a neighboring face via halo exchange. Even with "improved" ghost values (Putman & Lin Eq 47), the centered difference approach yields only **1st-order accuracy** at boundaries because:

1. Ghost cells represent data from a different coordinate system
2. The centered stencil crosses the panel boundary discontinuity
3. MC limiting further degrades accuracy near extrema

### The Solution

Ullrich et al. (2010) Appendix A specifies that at panel boundaries, derivatives **perpendicular to the edge** must use **one-sided stencils** that stay entirely within the current panel:

**Equation A.1 (Forward one-sided, 2nd-order):**
```
dq/dα|_i = (-3q_i + 4q_{i+1} - q_{i+2}) / (2Δα) + O(Δα²)
```

**Equation A.1 (Backward one-sided, 2nd-order):**
```
dq/dα|_i = (3q_i - 4q_{i-1} + q_{i-2}) / (2Δα) + O(Δα²)
```

These stencils:
- Use only interior cells (no ghost data for perpendicular derivative)
- Maintain 2nd-order accuracy
- Avoid crossing panel boundaries

---

## FV3s Scheme Specification

From Ullrich et al. (2010) Section 5.1, the FV3s (dimension-split piecewise-parabolic) scheme uses:

### Interior Derivative Stencils (Eq 48-49)

4th-order accurate, requires 5-point stencil:

```
dq/dα|_{i,j} = (-q_{i+2,j} + 8q_{i+1,j} - 8q_{i-1,j} + q_{i-2,j}) / (12Δ)

dq/dβ|_{i,j} = (-q_{i,j+2} + 8q_{i,j+1} - 8q_{i,j-1} + q_{i,j-2}) / (12Δ)
```

### Boundary Derivative Stencils (Eq A.1)

2nd-order accurate one-sided, requires 3-point stencil:

```
# At i=0 (left boundary), forward stencil:
dq/dα|_0 = (-3q_0 + 4q_1 - q_2) / (2Δ)

# At i=N-1 (right boundary), backward stencil:
dq/dα|_{N-1} = (3q_{N-1} - 4q_{N-2} + q_{N-3}) / (2Δ)
```

### Near-Boundary Cells

Cells at i=1 and i=N-2 cannot use the 5-point interior stencil. Use 3-point centered:

```
dq/dα|_i = (q_{i+1} - q_{i-1}) / (2Δ)
```

### No Slope Limiting

FV3s "does not explicitly limit the reconstructed derivatives" (Section 5.1). The MC limiter should be removed for the gradient computation.

**Note:** Slope limiting may still be needed for the advective flux reconstruction to maintain stability. This is a separate concern from the Bernoulli gradient.

---

## Implementation Plan

### Phase 1: Create `compute_slopes_fv3` Function

**File:** `Solvers/slope_reconstruction.py` (new file)

```python
def compute_slopes_fv3(f_interior, dx, N):
    """
    Compute slopes using FV3s stencils per Ullrich et al. (2010).
    
    - Interior (i=2 to N-3): 4th-order centered (Eq 48-49)
    - Near-boundary (i=1, N-2): 3-point centered
    - Boundary (i=0, N-1): One-sided 2nd-order (Eq A.1)
    
    Args:
        f_interior: (N, N) field values (NO ghost cells)
        dx: Grid spacing [radians]
        N: Grid size
    
    Returns:
        slope1, slope2: (N, N) slopes in ξ¹ and ξ² directions
    """
```

**Key design choice:** This function takes interior-only data `(N, N)` and does NOT require ghost cells, since boundary derivatives use one-sided stencils.

**Unit test:** `Tests/test_slope_reconstruction.py`
- Verify 4th-order convergence in interior
- Verify 2nd-order convergence at boundaries
- Verify against analytical gradients for polynomial test functions

### Phase 2: Create `compute_bernoulli_gradient_fv3` Function

**File:** `Solvers/swe_gradients.py` (new file)

```python
def compute_bernoulli_gradient_fv3(B, XI1, XI2, face_id, dx, N, R):
    """
    Compute ∇B in Cartesian coordinates using FV3s reconstruction.
    
    Args:
        B: (N, N) Bernoulli function [m²/s²]
        XI1, XI2: (N, N) cubed-sphere coordinates [radians]
        face_id: Cube face index (0-5)
        dx: Grid spacing [radians]
        N: Grid size
        R: Planet radius [m]
    
    Returns:
        dB_dX, dB_dY, dB_dZ: (N, N) Cartesian gradient components [m/s²]
    """
    # 1. Compute coordinate slopes using FV3s stencils
    slope1, slope2 = compute_slopes_fv3(B, dx, N)
    
    # 2. Transform to Cartesian via Jacobian
    # ∇B = J (J^T J)^{-1} [∂B/∂ξ¹, ∂B/∂ξ²]^T
    ...
```

**Unit test:** `Tests/test_swe_gradients.py`
- Test Case 2 geostrophic balance (the convergence study we already verified)
- Verify convergence rate ≈ 2.0

### Phase 3: Integrate into SWE Solver

**File:** `Solvers/fv_plr_cubesphere_swe.py`

#### 3a. Add imports

```python
from Solvers.slope_reconstruction import compute_slopes_fv3
from Solvers.swe_gradients import compute_bernoulli_gradient_fv3
```

#### 3b. Modify `compute_rhs` function

Replace the current Bernoulli gradient computation:

```python
# BEFORE (broken):
dB_dX, dB_dY, dB_dZ = compute_pressure_gradient_cartesian(
    B_ghosts / g, XI1_ghosts[face], XI2_ghosts[face],
    face, dx, N, g, R
)

# AFTER (fixed):
dB_dX, dB_dY, dB_dZ = compute_bernoulli_gradient_fv3(
    B[face], self.XI1_all[face], self.XI2_all[face],
    face, dx, N, R
)
```

**Note:** The new function does NOT need ghost cells for B, since one-sided stencils use only interior data.

#### 3c. Keep halo exchange for mass flux

The mass flux computation still needs halo exchange for the advective reconstruction. Only the Bernoulli gradient changes.

```python
def compute_rhs(h, Vx, Vy, Vz):
    # Halo exchange still needed for mass flux (advection)
    h_ghosts = exchange_scalar_halos_v2(h, N, self.halo_exchange)
    Vx_ghosts = exchange_scalar_halos_v2(Vx, N, self.halo_exchange)
    ...
    
    for face in range(6):
        # Mass equation: uses ghost cells for flux reconstruction
        mass_rhs = compute_mass_rhs(h_ghosts[face], ...)
        
        # Momentum equation: Bernoulli gradient uses FV3s (no ghosts)
        B = g * h[face] + 0.5 * (Vx[face]**2 + Vy[face]**2 + Vz[face]**2)
        dB_dX, dB_dY, dB_dZ = compute_bernoulli_gradient_fv3(
            B, XI1[face], XI2[face], face, dx, N, R
        )
```

### Phase 4: Validation

#### 4a. Test Case 2: Steady Geostrophic Flow

- Run at N = 15, 30, 60, 120
- Verify convergence rate ≈ 2.0
- Verify maximum error < 1% at N=60

#### 4b. Test Case 5: Zonal Flow over Mountain

- Verify no spurious oscillations
- Compare with published results

#### 4c. Long-term Integration

- Run Test Case 2 for 5+ days
- Verify mass conservation
- Verify energy/enstrophy behavior

---

## File Structure After Implementation

```
Solvers/
├── fv_plr_cubesphere_swe.py    # Main solver (modified)
├── slope_reconstruction.py      # NEW: FV3s slope stencils
├── swe_gradients.py            # NEW: Bernoulli gradient
├── halo_exchange.py            # Unchanged (still needed for flux)
└── halo_exchange_2deep.py      # May not be needed for this fix

Tests/
├── test_slope_reconstruction.py # NEW: Unit tests for slopes
├── test_swe_gradients.py       # NEW: Gradient convergence tests
└── test_swe_geostrophic.py     # Integration test (Test Case 2)
```

---

## Risk Assessment

### Low Risk
- The one-sided derivative approach is well-established (Ullrich 2010)
- We have verified 2.44 convergence rate in isolated testing
- Changes are localized to gradient computation

### Medium Risk
- Removing MC limiter may affect stability for non-smooth problems
- May need to retain limiting for advective flux while removing for Bernoulli gradient

### Mitigation
- Test with Test Case 5 (mountain) to verify stability
- Keep MC limiter as option, disabled by default for FV3s mode

---

## Timeline Estimate

| Phase | Task | Estimate |
|-------|------|----------|
| 1 | `compute_slopes_fv3` + unit tests | 2 hours |
| 2 | `compute_bernoulli_gradient_fv3` + tests | 1 hour |
| 3 | Integration into solver | 1 hour |
| 4 | Validation (Test Cases 2, 5) | 2 hours |
| **Total** | | **6 hours** |

---

## References

1. Ullrich, P.A., Jablonowski, C., and van Leer, B. (2010). "High-order finite-volume methods for the shallow-water equations on the sphere." J. Comp. Phys. 229: 6104-6134.
   - Section 5.1: FV3s scheme specification
   - Appendix A: One-sided boundary stencils (Eq A.1-A.4)
   - Eq 48-49: 4th-order interior stencils

2. Putman, W.M. and Lin, S.-J. (2007). "Finite-volume transport on various cubed-sphere grids." J. Comp. Phys. 227: 55-78.
   - Eq 47: Edge extrapolation (NOT the fix, but useful context)

3. Williamson, D.L. et al. (1992). "A standard test set for numerical approximations to the shallow water equations in spherical geometry." J. Comp. Phys. 102: 211-224.
   - Test Case 2: Steady geostrophic flow

---

## Appendix: Verified Test Results

From `test_geostrophic_balance_onesided.py`:

```
CONVERGENCE SUMMARY (ONE-SIDED BOUNDARY DERIVATIVES)
======================================================================

     N         dx      RMS Err   Expected dx²      Ratio
-------------------------------------------------------
    15     0.1047       0.54%        0.0110        0.5
    30     0.0524       0.10%        0.0027        0.4
    60     0.0262       0.02%        0.0007        0.3

  Observed convergence rate: 2.44
  Expected for 2nd-order: 2.0
```

The ratio < 1 indicates errors are **smaller** than the theoretical truncation error bound, which is excellent.
