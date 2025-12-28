# FV3 Shallow Water Equations (SWE) Implementation Status Report

**Date:** December 28, 2024  
**Version:** v0.3.0-alpha  
**Status:** Infrastructure complete, tensor formulation needs correction

---

## Executive Summary

The FV3 cubed-sphere shallow water equations solver has solid infrastructure (FV-PLR reconstruction, RK3 time integration, halo exchange, metric terms) but the momentum equations have a **tensor formulation error** that causes errors to grow with resolution refinement. The advection and diffusion solvers are production-ready; the SWE solver requires fixing the covariant/contravariant momentum formulation.

---

## ✅ What is WORKING

### 1. Core Infrastructure (Stable)

| Component | Demonstrated By | Status |
|-----------|-----------------|--------|
| FV-PLR reconstruction with MC limiter | `fv_plr_cubesphere_adv.py` | 2nd-order spatial ✓ |
| RK3 time integration | `fv_plr_cubesphere_swe.py` | 3rd-order temporal ✓ |
| Halo exchange across cube faces | `halo_exchange.py` | All 12 edges working ✓ |
| Metric terms (√G) in fluxes | All solvers | Properly included ✓ |
| Framework interface | `run_all_tests.py` | Interface validated ✓ |

### 2. Advection Solver (Production-Ready)

- **Test:** `Tests/test_validation.py` → `test_advection_exact_match`
- **Result:** Cosine bell advection, 12 days, N=120, matches reference to < 1e-10
- **Command:**
  ```bash
  python Tests/run_all_tests.py --full
  ```

### 3. Diffusion Solver (Production-Ready)

- **Test:** `Tests/test_validation.py` → `test_diffusion_exact_match`
- **Result:** Lima flag diffusion, 30 days, N=120, matches reference to < 1e-10
- **Command:**
  ```bash
  python Tests/run_all_tests.py --full
  ```

### 4. Continuity Equation with Prescribed Velocities

- **Test:** `Dev/swe_dev/test_swe_refactored.py --test-case testcase1 --prescribed-velocity`
- **Results:**
  - Mass conservation: ~4e-7 relative error after 50 steps ✓
  - Matches advection solver accuracy ✓

### 5. Vorticity/Coriolis Term Computation

- **Test:** `Dev/swe_dev/diagnostic_plots.py` and `test_vorticity.py`
- **Results:**
  - Uses analytical absolute vorticity for solid body rotation:
    `η = 2(Ω + u₀/R) · sin(lat)` (Eq 94 from Williamson et al.)
  - Matches analytical to machine precision at t=0
  
  ```
  Vorticity term (at t=0):
    Max error: 8.67e-19 m/s²
    Relative: 2.73e-16
  ```

### 6. Bernoulli Function

- Properly computes `B = gh + K` where `K = |V|²/2`
- Gradient computed correctly in interior cells

---

## ❌ What is NOT WORKING

### 1. Geostrophic Balance at Cube Edges (~20-50% imbalance)

- **Test:** `Dev/swe_dev/test_geostrophic_balance.py`, `diagnostic_plots.py`
- **Evidence:**
  ```
  Geostrophic imbalance (at t=0):
    Max absolute: 7.30e-04 m/s²
    Max relative: 27.5%
    Location: Face 1, near corner
  ```
- **Root Cause:** Pressure gradient only 1st-order accurate at cube edges (uses 1-deep ghost cells)
- **Reference:** Putman & Lin (2007) note that cubed-sphere corners have 120° coordinate intersections, causing geometric singularities.

### 2. Test Case 2 Convergence (WRONG SIGN)

- **Test:** `Dev/swe_dev/convergence_study.py`
- **Evidence:**
  ```
  N      dx(km)     dt(s)      L2(h)        L2(u1)       L2(u2)       Mass Err    
  ----------------------------------------------------------------------
  30     333.58     416.87     1.25e+02     4.74e+00     7.82e+00     7.91e-04    
  60     166.79     208.43     2.88e+02     5.14e+00     1.42e+01     4.18e-03    
  120    83.40      104.22     3.86e+02     6.01e+00     3.25e+01     6.12e-03    

  Convergence Rates:
    N=30 → N=60:  Rate(h) = -1.21 (WRONG SIGN - errors increasing!)
    N=60 → N=120: Rate(h) = -0.42 (WRONG SIGN - errors increasing!)
  ```
- **Interpretation:** Errors grow with refinement → formulation bug, not discretization error

### 3. Mass & Energy Drift in Full SWE

- **Test:** `Dev/swe_dev/test_5_steps.py` → output files `test_5_steps.n*.out`
- **Results after 5 steps (N=30):**
  ```
  Step 5 (t=2500s):
    Mass error:   9.41e-04 (growing)
    Energy error: 1.17e-03 (growing)
  ```

### 4. Putman & Lin Edge Extrapolation (Not Improving)

- **Test:** `Dev/swe_dev/test_edge_extrapolation.py`
- **Evidence:**
  ```
  RMS imbalance:
    Standard:   1.7834e-04 m/s²
    Edge-aware: 3.5855e-03 m/s²
    Improvement: -1910.5%   ← WORSE!
  ```
- **Status:** Implementation attempted but not producing expected improvements; may need debugging

---

## 🔍 Key Diagnostic Tests

| Test File | What It Checks | Current Status |
|-----------|----------------|----------------|
| `test_geostrophic_balance.py` | g∇h = f×V | ❌ 27.5% imbalance at edges |
| `test_flux_divergence_simple.py` | ∇·V = 0 for solid rotation | ✓ Machine precision globally |
| `test_vorticity.py` | η = ζ + f analytical match | ✓ 2.7e-16 relative error |
| `convergence_study.py` | 2nd-order in smooth regions | ❌ Negative convergence rate |
| `test_5_steps.py` | Short integration stability | ⚠ Stable but drifting |
| `diagnostic_plots.py` | Visual diagnostics | Generates error maps |

### Running Diagnostics

```bash
cd Dev/swe_dev
source /path/to/jax_venv/bin/activate

# Geostrophic balance check
python test_geostrophic_balance.py

# Vorticity check
python test_vorticity.py

# Generate diagnostic plots
python diagnostic_plots.py

# Test edge extrapolation
python test_edge_extrapolation.py

# Short integration test
python test_5_steps.py
```

---

## 📋 Root Cause Analysis

### The Problem

We're using a **naive flux-form** that mixes covariant and contravariant components:

**Current (WRONG):**
```
∂h/∂t = -(1/(R√G)) [∂(hu¹√G)/∂ξ¹ + ∂(hu²√G)/∂ξ²]
∂(hu¹)/∂t = -(1/(R√G)) ∂[(hu¹·u¹ + 0.5gh²)√G]/∂ξ¹ + f·hu²
∂(hu²)/∂t = -(1/(R√G)) ∂[(hu²·u² + 0.5gh²)√G]/∂ξ² - f·hu¹
```

**Correct (Nair et al. 2005):**
```
∂h/∂t + ∂/∂x^i (√G u^i h) = 0

∂u_i/∂t + ∂E/∂x^i = -√G u^j (f + ζ) ε_{ij}

where:
  u_i = G_{ij} u^j           (covariant ← contravariant)
  E = Φ + (1/2)(u_1 u^1 + u_2 u^2)  (Bernoulli function)
  ζ = (1/√G)[∂u_2/∂x^1 - ∂u_1/∂x^2]  (relative vorticity)
  G_{ij} = metric tensor for equiangular cubed sphere
```

### Key Differences

1. **Integrate covariant momentum** `u_i`, not contravariant `u^i`
2. **Coriolis uses Levi-Civita tensor** `ε_{ij}` and includes vorticity
3. **Bernoulli form** combines kinetic energy + geopotential
4. **Proper metric tensor** `G_{ij}` for coordinate transformations

---

## 🛣️ Path Forward

### Phase 1: Correct Formulation

1. **Implement metric tensor** `G_{ij}` for equiangular cubed sphere
   - From Nair et al.: `G_{ij} = 1/(r^4 cos^2 x^1 cos^2 x^2) * [[...]]`
   - Add to `CubedSphereGeometry` class

2. **Transform initial conditions**
   - Convert `(u_lon, u_lat)` → contravariant `(u^1, u^2)`
   - Then compute covariant `(u_1, u_2) = G_{ij} u^j`
   - Initialize with `(h, hu_1, hu_2)` not `(h, hu^1, hu^2)`

3. **Rewrite momentum equations**
   - Compute Bernoulli function `E`
   - Compute relative vorticity `ζ`
   - Add Coriolis-vorticity term: `-√G u^j (f + ζ) ε_{ij}`

4. **Update flux computation**
   - Reconstruct covariant `u_i` at cell centers
   - Convert to contravariant `u^i` for flux divergence
   - Proper tensor index gymnastics throughout

### Phase 2: Validation

1. Run Test Case 2 with corrected formulation
2. Verify convergence rate ≈ 2.0 for smooth flow
3. Add Test Case 5 (zonal flow over mountain)
4. Add Test Case 6 (Rossby-Haurwitz wave)
5. Add Galewsky barotropic instability

### Phase 3: Production

1. Add proper documentation in `Solvers/fv_plr_cubesphere_swe.py`
2. Create reference solutions for regression testing
3. Integrate with formal test suite
4. Performance optimization (JIT compilation, GPU)

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `Solvers/fv_plr_cubesphere_swe.py` | Main SWE solver (needs tensor fix) |
| `Solvers/geometry/cubesphere.py` | Geometry (add G_ij metric tensor here) |
| `Solvers/initial_conditions/swe_testcase2.py` | IC (needs u_i transform) |
| `Solvers/physics/planet.py` | Coriolis parameter f(φ) |
| `Dev/swe_dev/IMPLEMENTATION_STATUS.md` | Current dev status |
| `Dev/swe_dev/edge_extrapolation.py` | Putman & Lin edge algorithms |
| `Dev/swe_dev/halo_exchange_2deep.py` | 2-deep halo exchange |
| `Tests/run_all_tests.py` | Main test suite |
| `Tests/test_swe_testcase2.py` | SWE test template |

---

## 📚 References

1. **Nair, R. D., Thomas, S. J., & Loft, R. D. (2005).** "A discontinuous Galerkin transport scheme on the cubed sphere." *Monthly Weather Review*, 133(4), 814-828.
   - See equations (3)-(5) for proper tensor formulation
   - See Section 2b for metric tensor on equiangular cubed sphere

2. **Putman, W.M. and Lin, S.-J. (2007).** "Finite-volume transport on various cubed-sphere grids." *J. Comp. Phys.* 227: 55-78.
   - Equations 47, 49 for 3rd-order edge extrapolation

3. **Williamson, D. L., et al. (1992).** "A standard test set for numerical approximations to the shallow water equations in spherical geometry." *Journal of Computational Physics*, 102(1), 211-224.
   - Test Case 2: Steady geostrophic zonal flow
   - Acceptance criteria for L1, L2, L∞ norms

---

## 🎯 Summary

| Category | Status |
|----------|--------|
| **Advection Solver** | ✅ Production-ready |
| **Diffusion Solver** | ✅ Production-ready |
| **SWE Continuity (prescribed V)** | ✅ Working |
| **SWE Vorticity Term** | ✅ Machine precision |
| **SWE Momentum Equations** | ❌ Wrong tensor formulation |
| **SWE Geostrophic Balance** | ❌ 27% imbalance at edges |
| **SWE Convergence Rate** | ❌ Negative (errors grow with resolution) |

**Bottom line:** The solver infrastructure is solid (PLR, RK3, halos, metrics all working), but the **tensor formulation is wrong**. This is a **physics bug**, not a numerical bug. Fix requires careful implementation of covariant/contravariant transformations using the proper metric tensor.

**Estimated effort for fix:** 4-6 hours of careful implementation + testing.

