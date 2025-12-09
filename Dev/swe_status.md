# Shallow Water Equations (SWE) Solver - Status Report

**Date:** December 8, 2024  
**Version:** v0.3.0-alpha  
**Status:** Prototype complete, needs proper tensor formulation

---

## Current Implementation

### What Works ✅
- **FV-PLR reconstruction** with MC limiter (2nd order spatial)
- **RK3 time integration** (3rd order temporal)
- **Conservative flux-form** discretization
- **Halo exchange** working correctly for all fields
- **Metric terms** (√G) properly included in fluxes
- **Solver compiles and runs stably** for 3600+ timesteps
- **Mass conservation** ~0.1-0.6% over 1 hour
- **Energy conservation** ~0.4-0.6% over 1 hour

### What Doesn't Work ❌
- **Coriolis term** implemented incorrectly
  - Current: naive `f·hu²` and `-f·hu¹`
  - Correct: `-√G u^j (f + ζ)` in Bernoulli form (Nair et al. 2005)
- **Covariant vs contravariant confusion**
  - Treating `(hu1, hu2)` as both contravariant AND covariant
  - Should integrate covariant momentum `(hu_1, hu_2)`
  - Fluxes use contravariant velocity `(u^1, u^2)`
- **Initial conditions mismatch**
  - `steady_geostrophic_flow()` returns `(u_lon, u_lat)`
  - Need proper transformation to contravariant `(u^1, u^2)` on cubed sphere
- **Test Case 2 diverging**
  - Errors INCREASE with resolution (should decrease)
  - Indicates fundamental formulation error, not discretization error

---

## Test Results

### Latest Convergence Study
**Script:** `Dev/swe_dev/convergence_study.py`

**Command:**
```bash
cd /Users/loft/Desktop/AreandDee/Clients/Google-CSU/Wraptor
source /path/to/jax_venv/bin/activate
python Dev/swe_dev/convergence_study.py
```

**Results (1 hour integration, CFL=0.4):**
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

**Interpretation:**
- Errors grow with refinement → formulation bug, not discretization error
- Root cause: incorrect tensor formulation (covariant vs contravariant)

### Other Dev Scripts
- `Dev/swe_dev/test_5_steps.py` - Quick 5-step stability test
- `Dev/swe_dev/diagnose_rhs.py` - Check CFL and RHS magnitudes
- `Dev/swe_dev/visualize_swe_initial.py` - Plot initial conditions (h, u1, u2)

---

## Root Cause Analysis

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

## Path Forward

### Phase 1: Correct Formulation (Next Session)
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

## References

- **Nair, R. D., Thomas, S. J., & Loft, R. D. (2005).** "A discontinuous Galerkin transport scheme on the cubed sphere." *Monthly Weather Review*, 133(4), 814-828.
  - See equations (3)-(5) for proper tensor formulation
  - See Section 2b for metric tensor on equiangular cubed sphere
  
- **Williamson, D. L., et al. (1992).** "A standard test set for numerical approximations to the shallow water equations in spherical geometry." *Journal of Computational Physics*, 102(1), 211-224.
  - Test Case 2: Steady geostrophic zonal flow
  - Acceptance criteria for L1, L2, L∞ norms

---

## Current File Structure

```
Wraptor/
├── Solvers/
│   ├── fv_plr_cubesphere_swe.py      # Main SWE solver (needs rewrite)
│   ├── geometry/
│   │   └── cubesphere.py             # Add G_ij metric tensor here
│   ├── initial_conditions/
│   │   └── swe_testcase2.py          # Working (but needs u_i transform)
│   └── physics/
│       └── planet.py                 # Coriolis parameter f(φ)
├── Dev/
│   ├── swe_design.md                 # Original design doc
│   ├── swe_status.md                 # THIS FILE
│   └── swe_dev/                      # Dev scripts (not committed)
│       ├── convergence_study.py      # Latest convergence results
│       ├── test_5_steps.py
│       ├── diagnose_rhs.py
│       └── visualize_swe_initial.py
├── Config/
│   └── swe_framework.yaml            # Config for SWE runs
└── Tests/
    └── test_swe_testcase2.py         # Template (not used yet)
```

---

## Commit Message (Suggested)

```
v0.3.0-alpha: FV-PLR RK3 SWE prototype (needs tensor formulation fix)

Implemented:
- FV-PLR with MC limiter (2nd order spatial)
- RK3 time integration (3rd order temporal)
- Conservative flux-form with metric terms
- Proper halo exchange at each RK stage
- Stable for 3600+ timesteps
- ~0.5% mass/energy conservation

Known Issues:
- Covariant vs contravariant components confused
- Coriolis term incorrectly formulated
- Test Case 2 errors INCREASE with refinement (not converging)
- Need proper tensor formulation from Nair et al. 2005

Next: Implement correct G_ij metric tensor and Bernoulli-form momentum equations

See Dev/swe_status.md for full analysis and path forward.
```

---

## Summary

**Bottom line:** The solver infrastructure is solid (PLR, RK3, halos, metrics all working), but the **tensor formulation is wrong**. This is a **physics bug**, not a numerical bug. Fix requires careful implementation of covariant/contravariant transformations using the proper metric tensor.

**Estimated effort for fix:** 4-6 hours of careful implementation + testing.

**Current state is useful for:** Demonstrating framework architecture, showing solver stability, testing I/O and sharding infrastructure.

