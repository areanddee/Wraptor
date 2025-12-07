# Flux-Form Shallow Water Equations on Cubed Sphere

## Coordinate System
- Computational coordinates: `ξ¹, ξ²` (dimensionless, typically radians on unit sphere)
- Grid spacing: `dξ = dx` (in radians)
- Physical sphere radius: `R` [m]
- Physical spacing: `Δx_physical = R · dx` [m]

## State Variables
- `h(ξ¹, ξ²)` : fluid depth [m]
- `u¹(ξ¹, ξ²)` : contravariant velocity in ξ¹ direction [m/s]
- `u²(ξ¹, ξ²)` : contravariant velocity in ξ² direction [m/s]

## Metric Terms
- `√G(ξ¹, ξ²)` : Jacobian of coordinate transformation (unitless)
- For unit sphere: `√G = f(ξ¹, ξ²)` depends on projection
- For physical sphere of radius R: scale by R²? NO - velocities already in [m/s]

## Conservation Form

### Mass Conservation:
```
∂h/∂t + (1/√G) · (1/R) · [∂(h·u¹·√G)/∂ξ¹ + ∂(h·u²·√G)/∂ξ²] = 0
```

**Units check:**
- LHS: `[m]/[s] = [m/s]`
- RHS: `(1/√G) · (1/R) · [∂(m·(m/s)·√G)/∂(radians)]`
  - `= (1/√G) · (1/[m]) · [m·m/s·√G] / [radians]`
  - `= (1/[m]) · [m²/s]`
  - `= [m/s]` ✓

### Momentum Conservation (ξ¹ direction):
```
∂(h·u¹)/∂t + (1/√G) · (1/R) · ∂[(h·u¹·u¹ + 0.5·g·h²)·√G]/∂ξ¹ + ... = Coriolis + other
```

**Units check:**
- LHS: `[m·m/s]/[s] = [m²/s²]`
- RHS pressure term: `(1/√G) · (1/R) · ∂[(g·h²)·√G]/∂ξ¹`
  - `= (1/√G) · (1/[m]) · [m/s²·m²·√G]`
  - `= (1/[m]) · [m³/s²]`
  - `= [m²/s²]` ✓

### Momentum Conservation (ξ² direction):
```
∂(h·u²)/∂t + (1/√G) · (1/R) · ∂[(h·u²·u² + 0.5·g·h²)·√G]/∂ξ² + ... = Coriolis + other
```

## Finite Difference Implementation

For a uniform grid with spacing `dx` (in radians):

```python
# Scale factor for converting computational derivatives to physical
scale = 1.0 / (R * dx)  # [1/m] / [radians] = [1/m]

# Mass conservation
dh_dt = -scale * (1.0 / sqrtG) * (
    jnp.gradient(h * u1 * sqrtG, axis=1) +  # ∂(h·u¹·√G)/∂ξ¹
    jnp.gradient(h * u2 * sqrtG, axis=2)    # ∂(h·u²·√G)/∂ξ²
)

# Momentum (ξ¹ direction)
flux1 = (h * u1 * u1 + 0.5 * g * h * h) * sqrtG
dhu1_dt = -scale * (1.0 / sqrtG) * jnp.gradient(flux1, axis=1)

# Momentum (ξ² direction)  
flux2 = (h * u2 * u2 + 0.5 * g * h * h) * sqrtG
dhu2_dt = -scale * (1.0 / sqrtG) * jnp.gradient(flux2, axis=2)
```

**Note:** `jnp.gradient()` computes `∂f/∂index` where index spacing = 1, so we need to account for `dx`.

## Why u² tendency is non-zero in current buggy code

Test Case 2 has:
- `u² = 0` everywhere (zonal flow)
- `h(λ, φ) = h₀ - (1/g)·(a·Ω·u₀ + u₀²/2)·sin²(φ)` (varies with latitude)

The pressure gradient in ξ² direction:
```
∂(0.5·g·h²)/∂ξ² = g·h·(∂h/∂ξ²) ≠ 0
```

This is **physically correct** - the pressure gradient balances the Coriolis force to maintain geostrophic equilibrium. However, our simplified test is missing the Coriolis term, so it will drift!

## Action Items

1. ✅ Add `scale = 1/(R*dx)` factor
2. ✅ Multiply mass/momentum fluxes by `√G` before differencing
3. ✅ Divide result by `√G` after differencing
4. ✅ Add halo exchange before computing gradients
5. ⚠️  For Test Case 2, either:
   - Add Coriolis terms to balance pressure gradient
   - OR accept small drift (this is a known limitation of simplified tests)

