# FV-PLR Design for Flux-Form Shallow Water Equations

## 1. Flux-form SWE on the Rotating Cubed Sphere
The conservative state vector is
```
U = [h, (hu)^{ξ1}, (hu)^{ξ2}]
```
where `h` is fluid depth and `(hu)^{ξi}` are contravariant momenta along the equiangular coordinates `ξ¹`, `ξ²`. The flux-form equations on a rotating sphere (radius `R`, rotation rate `Ω`) are:
```
∂ₜh + ∇_ξ · (h V) = 0
∂ₜ(hV) + ∇_ξ · (h V ⊗ V + ½ g h² G⁻¹) + f k̂ × (h V) = S_geo
```
with `V = (u^{ξ1}, u^{ξ2})`, `g` gravity, `G` the metric tensor derived from the equiangular-to-Cartesian map, and `f = 2Ω sin(φ)` the Coriolis parameter (`φ` = latitude). The source `S_geo` collects geometric (metric) forcing and centrifugal terms that arise from the curvilinear coordinate system.

### Discretization highlights
- Finite volumes reside on each tile (face) cell; the metric determinant `√G(ξ¹, ξ²)` comes from `Solvers.geometry.CubedSphereGeometry` so the same grids used in diffusion/advection apply.
- PLR reconstruction is performed on each conservative component. The MC slope limiter is evaluated using inward neighbor values, including ghost cells obtained via `halo_exchange` across face boundaries.
- Edge fluxes use contravariant velocities projected along the tile normals. Celebrate the existing halo-exchange schedule—embedding our flux computation inside the exchange routine ensures consistent ordering of neighbor tiles.
- Metric correction terms (from derivatives of the mapping) are added as point-wise sources so steady geostrophic solutions remain equilibrated.

## 2. Halo exchange + FV-PLR synergy
- The previously implemented halo exchange already permutes scalar fields across cube faces with the correct orientation bookkeeping. FV-PLR reuses these ghost values when computing interface slopes and fluxes across tiles.
- Even when `tiles_per_edge = 1`, the halo exchange handles the `ξ¹/ξ²` rotation of the faces, so slope reconstructions remain continuous across the full sphere.
- This decoupling means each solver stays agnostic about the comms wiring: we just request `state_with_halos = halo_exchange(state)` and compute fluxes locally.

## 3. Test Case 2 (Williamson, 1992)
Test Case 2 is a **steady geostrophic zonal flow** defined by
```
h(λ, φ) = h₀ - (a Ω u₀ / g) × (a Ω cos λ (2 sin² φ - cos² φ))
```
with constant zonal velocity `u₀` and `λ` longitude. Physical constants:
- Radius `a = 6.37122 × 10⁶ m`
- Rotation `Ω = 7.292115 × 10⁻⁵ s⁻¹`
- Gravity `g = 9.80616 m/s²`
- Mean depth `h₀ = 8000 m`

The steady analytic solution satisfies `∂ₜh = 0` and `∂ₜV = 0` since the pressure gradient exactly balances Coriolis. Implementation steps:
1. Initialize `h` and `V` from the analytic expressions on the cubed-sphere grid.
2. Run the FV-PLR solver for `t_final = 5` days (or longer) using the new geometry modules.
3. Confirm no spurious drift: the zonal flow should hold constant.

## 4. Error tracking & convergence methodology
For EVERY snapshot or output frame:
- Compute deviations `Δh = h_num - h_analytic`, `ΔV = V_num - V_analytic`.
- Evaluate norms:
  - `L₁ = (1/N_tot) ∑ |Δ|`
  - `L₂ = sqrt((1/N_tot) ∑ Δ²)`
  - `L_∞ = max |Δ|`
  where `N_tot = 6 × N² × (#fields)` and sums include metric weights when appropriate.
- Track invariants: mass `∑ h √G`, energy `∑ (½ h |V|² + ½ g h²) √G`, and potential vorticity to spot drift early.

### Convergence plan
- Run the test with `N = 30, 60, 120`. For each run, log the three norms at `t_final` and compute the empirical convergence rate `r = log(error₂/error₁)/log(dx₂/dx₁)`.
- Verify PLR approximations show near-second-order accuracy (r ≈ 2) for smooth flows.
- Acceptance criteria: at the highest resolution, expect `L_∞(h)` < 1e-3, `L₂(h)` < 1e-4, and relative changes in mass/energy < 1e-6 (ensuring well-balanced behavior).

## 5. Test harness considerations
- The precision test infrastructure can be extended to compute SWE norms instead of diffusion/advection diagnostics.
- Add regression references (e.g., final `h` fields) for the steady case to the `Tests/validation` directory once reliability is achieved.
- Use the existing `IOManager` for history output, letting config define which fields to record (height, momentum, diagnostics).

