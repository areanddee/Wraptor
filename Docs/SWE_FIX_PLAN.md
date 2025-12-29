# SWE Solver Fix Plan

**Date:** December 28, 2024  
**Status:** Diagnosis Complete, Fix Ready to Implement

---

## Executive Summary

The SWE solver has a **velocity coordinate transformation bug** that causes errors to grow with resolution. The fix is straightforward: adopt the same Cartesian velocity storage pattern used by the working advection solver.

---

## Root Cause Analysis

### The Bug

In `Solvers/fv_plr_cubesphere_swe.py`, lines 80-81:

```python
hu1 = jnp.array(h * u_lon)  # WRONG: u_lon is spherical, not cubed-sphere
hu2 = jnp.array(h * u_lat)  # WRONG: u_lat is spherical, not cubed-sphere
```

The initial conditions return `(u_lon, u_lat)` in **spherical coordinates** (eastward, northward), but the solver treats them as cubed-sphere contravariant coordinates `(u¹, u²)`.

### Why This Causes Negative Convergence

- `u_lon ≠ u¹` except at special locations where cubed-sphere axes happen to align with east/north
- The discrepancy is **geometric**, not numerical
- Refining the grid doesn't fix a geometry error—it just samples the wrong velocity field more finely
- Result: errors **grow** with resolution instead of shrinking

### Why Advection Works

The advection solver (`fv_plr_cubesphere_adv.py`) stores velocity in **Cartesian form** `(Vx, Vy, Vz)`:
- Cartesian is globally consistent (no coordinate singularities)
- Conversion to contravariant `(u¹, u²)` happens at flux computation time
- The `cartesian_to_contravariant()` function handles the proper Jacobian transform

---

## Recommended Fix: Cartesian Momentum Storage

### Step 1: Change SWE State Structure

**Before:**
```python
@struct.dataclass
class SWEState(SolverState):
    h: jnp.ndarray      # (6, N, N) depth
    hu1: jnp.ndarray    # (6, N, N) momentum in ξ¹ direction - WRONG
    hu2: jnp.ndarray    # (6, N, N) momentum in ξ² direction - WRONG
    time: float
    step: int
```

**After:**
```python
@struct.dataclass
class SWEState(SolverState):
    h: jnp.ndarray      # (6, N, N) depth
    hVx: jnp.ndarray    # (6, N, N) x-momentum (Cartesian)
    hVy: jnp.ndarray    # (6, N, N) y-momentum (Cartesian)
    hVz: jnp.ndarray    # (6, N, N) z-momentum (Cartesian)
    time: float
    step: int
```

### Step 2: Fix Initialization

**Before:**
```python
def initialize(self, config):
    h, u_lon, u_lat = steady_geostrophic_flow(...)
    hu1 = h * u_lon  # WRONG
    hu2 = h * u_lat  # WRONG
    return SWEState(h=h, hu1=hu1, hu2=hu2, ...)
```

**After:**
```python
def initialize(self, config):
    h, u_lon, u_lat = steady_geostrophic_flow(...)
    
    # Convert spherical → Cartesian
    lat, lon = self.geometry.get_lat_lon_all_faces()
    Vx, Vy, Vz = spherical_to_cartesian_velocity(u_lon, u_lat, lon, lat)
    
    hVx = h * Vx
    hVy = h * Vy
    hVz = h * Vz
    
    return SWEState(h=h, hVx=hVx, hVy=hVy, hVz=hVz, ...)

def spherical_to_cartesian_velocity(u_lon, u_lat, lon, lat):
    """Transform (u_lon, u_lat) to (Vx, Vy, Vz)."""
    sin_lon, cos_lon = jnp.sin(lon), jnp.cos(lon)
    sin_lat, cos_lat = jnp.sin(lat), jnp.cos(lat)
    
    Vx = -u_lon * sin_lon - u_lat * sin_lat * cos_lon
    Vy = u_lon * cos_lon - u_lat * sin_lat * sin_lon
    Vz = u_lat * cos_lat
    
    return Vx, Vy, Vz
```

### Step 3: Fix Flux Computation

At flux time, convert Cartesian → contravariant:

```python
def compute_rhs(self, h, hVx, hVy, hVz):
    # Convert Cartesian momentum to contravariant for flux divergence
    Vx, Vy, Vz = hVx/h, hVy/h, hVz/h
    
    for face in range(6):
        # Convert to contravariant at this face
        u1, u2 = cartesian_to_contravariant(
            Vx[face], Vy[face], Vz[face],
            self.XI1_all[face], self.XI2_all[face], face
        )
        hu1 = h[face] * u1
        hu2 = h[face] * u2
        
        # Compute fluxes using (hu1, hu2) ...
```

### Step 4: Fix Coriolis Term

The Coriolis force is naturally in Cartesian:

```python
def compute_coriolis_cartesian(Vx, Vy, Vz, X, Y, Z, omega):
    """Coriolis acceleration: -2Ω × V in Cartesian."""
    # Ω = omega * (0, 0, 1) for Earth
    # -2Ω × V = -2*omega * (Vy, -Vx, 0)... but on sphere
    
    # For sphere: Ω_local = omega * r_hat * sin(lat)
    # More precisely: -2Ω × V where Ω is Earth's rotation
    
    # Result is tangent to sphere
    cor_x = 2 * omega * Vy * Z  # Simplified for testing
    cor_y = -2 * omega * Vx * Z
    cor_z = 2 * omega * (Vx * Y - Vy * X)
    
    return cor_x, cor_y, cor_z
```

### Step 5: Add Metric Source Terms

For proper momentum conservation in curvilinear coordinates:

```python
def compute_metric_source(self, hu1, hu2, face):
    """Christoffel symbol terms (Ullrich et al. 2010, Eq. 15)."""
    X = jnp.tan(self.XI1_all[face])
    Y = jnp.tan(self.XI2_all[face])
    d2 = 1 + X**2 + Y**2
    
    u1 = hu1 / h
    u2 = hu2 / h
    
    W_M1 = (2/d2) * (X*Y**2*hu1*u1 + Y*(1+Y**2)*hu1*u2)
    W_M2 = (2/d2) * (X*(1+X**2)*hu1*u2 - X**2*Y*hu2*u2)
    
    return W_M1, W_M2
```

---

## Incremental Implementation Steps

### Phase 1: Verify Diagnosis (1 hour)
1. Run `pytest Tests/test_swe_unit_phase2_velocity.py -v -s`
2. Confirm diagnostic output shows spherical ≠ contravariant
3. Document baseline convergence rate (should be negative)

### Phase 2: Add Cartesian Storage (2 hours)
1. Modify `SWEState` dataclass
2. Add `spherical_to_cartesian_velocity()` function
3. Update `initialize()` method
4. Run Phase 1 geometry tests to verify no regression

### Phase 3: Update Flux Computation (2 hours)
1. Add `cartesian_to_contravariant()` calls in `compute_rhs()`
2. Ensure halo exchange works with Cartesian fields
3. Run Phase 3 flux tests

### Phase 4: Add Source Terms (1 hour)
1. Implement `compute_metric_source()`
2. Update Coriolis to use Cartesian velocities
3. Run Phase 4 source tests

### Phase 5: Validate (2 hours)
1. Run full Test Case 2
2. Measure convergence rate (should be ~2.0)
3. Run Phase 5 integration tests
4. Generate reference solutions for regression testing

---

## Test Suite Summary

| File | Purpose | Status |
|------|---------|--------|
| `test_swe_unit_phase1_geometry.py` | Coordinate transforms, metric | ✅ Ready |
| `test_swe_unit_phase2_velocity.py` | Velocity transforms (documents bug) | ✅ Ready |
| `test_swe_unit_phase3_flux.py` | Flux computation, halo exchange | ✅ Ready |
| `test_swe_unit_phase4_sources.py` | Coriolis, metric source | ✅ Ready |
| `test_swe_unit_phase5_integration.py` | Full solver, convergence | ✅ Ready |

Run all with:
```bash
pytest Tests/test_swe_unit_phase*.py -v
```

---

## Success Criteria

After implementing the fix:

1. **Convergence rate ≈ 2.0** for Test Case 2 (currently negative)
2. **Mass conserved** to machine precision
3. **Geostrophic balance** maintained (< 1% imbalance)
4. **All unit tests pass**

---

## References

1. Ullrich et al. (2010) - Equations 3-19 for metric terms
2. Nair et al. (2005) - DG formulation with proper tensor indices
3. Williamson et al. (1992) - Test Case 2 specification
4. `Solvers/fv_plr_cubesphere_adv.py` - Working reference implementation

