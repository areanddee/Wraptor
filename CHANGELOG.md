# Wraptor Changelog

## v0.2.0 - Framework Refactor & SWE Prototype (2024-12-07)

### Framework
- ✅ Formal `NumericalSolver` abstract base class with 7 required methods
- ✅ Generic `IOManager` for Zarr history output and Orbax checkpointing
- ✅ Unified `runner.py` for all simulations
- ✅ Solver-driven I/O: solvers define what/when to save, framework handles mechanics
- ✅ YAML config system for physics parameters, time integration, and I/O specs

### Geometry & Physics Extraction
- ✅ `CubedSphereGeometry` module (unit sphere math, metric terms)
- ✅ `PlanetParams` module (R_sphere, gravity, omega, Coriolis)
- ✅ Precision-agnostic design (FP32/FP64 configurable)

### Initial Conditions
- ✅ `lima_flag.py` (thermal diffusion test)
- ✅ `cosine_bell.py` + `solid_body_rotation.py` (advection test)
- ✅ `swe_testcase2.py` (steady geostrophic flow) - **FIXED formula**

### Solvers
- ✅ `fv_cubesphere_diffusion.py` - refactored to use new framework
- ✅ `fv_plr_cubesphere_adv.py` - refactored to use new framework
- ⚠️ `fv_plr_cubesphere_swe.py` - **PROTOTYPE ONLY** (compiles, but unstable)
  - Uses crude 2nd-order centered differences (not true FV-PLR)
  - Forward Euler timestepping (not RK3)
  - Missing full metric terms
  - **Deferred:** Proper FV-PLR RK3 implementation (next phase)

### Testing
- ✅ `test_precision.py` (FP32 vs FP64 mass conservation)
- ✅ `test_validation.py` (regression tests vs gold standards)
- ✅ Pytest dashboard runner with sharding options
- ✅ Reference solution generation/verification

### Visualization
- ✅ `Analysis/visualize_sphere.py` (2D lat-lon + movie generation)
- ✅ Linear interpolation to avoid overshoot artifacts
- ✅ Progress bars and `--skip-first-frame` option
- ✅ ffmpeg even-dimension safeguards

### Documentation
- ✅ `Dev/swe_design.md` - flux-form SWE equations and test case design
- ✅ `Dev/README.md` - development workflow guidelines
- ✅ `Analysis/README.md` - presentation workflow documentation
- ✅ `.gitignore` updates for dev outputs

### Bug Fixes
- Fixed SWE Test Case 2 initial condition formula (removed spurious longitude dependence)
- Fixed geometry coordinate tuple unpacking in visualization
- Fixed scale factors in SWE flux divergence (R·dx conversion)
- Fixed halo exchange integration (proper ghosted array handling)

### Known Issues
- SWE solver unstable beyond ~40 steps (needs proper FV-PLR RK3)
- Missing Coriolis force in momentum equations
- Energy not conserved (nonlinear advection instability)

### Next Phase (v0.3.0)
- [ ] Implement proper FV-PLR RK3 for SWE (adapt from advection solver template)
- [ ] Add Coriolis force and full metric terms
- [ ] Run convergence study (N=30, 60, 120)
- [ ] Compute L1/L2/L∞ error norms vs analytic solution
- [ ] Add more Williamson test cases (TC5, TC6, Galewsky)

