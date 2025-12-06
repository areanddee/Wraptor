# JaxStream2 Testing Infrastructure

This directory contains validation tests and reference solutions to ensure code changes don't break physics correctness.

## Structure

```
Tests/
├── validation/                           # Reference "gold standard" solutions
│   ├── diffusion_lima_flag_day30_N120.npz
│   └── advection_cosine_bell_day12_N120.npz
├── generate_reference_solutions.py      # Generate new reference data
├── verify_references.py                 # Quick check that references exist
├── test_validation.py                   # Regression tests (pytest compatible)
└── README.md                            # This file
```

## Reference Solutions

Reference solutions are "gold standard" outputs used for regression testing. They capture the expected behavior of working code.

### Current References

| Test Case | Solver | Resolution | Duration | File |
|-----------|--------|------------|----------|------|
| Lima Flag | Diffusion | N=120 | 30 days | `diffusion_lima_flag_day30_N120.npz` |
| Cosine Bell | Advection | N=120 | 12 days | `advection_cosine_bell_day12_N120.npz` |

### When to Update References

**ONLY update reference solutions when you intentionally change physics:**
- ✅ Fixed a numerical bug
- ✅ Improved accuracy (better limiter, higher-order method)
- ✅ Changed test parameters (different resolution, duration)

**DO NOT update references to "fix" failing tests** — fix the code instead!

## Running Tests

### 🎯 Test Dashboard (RECOMMENDED)
```bash
# Quick tests only (~50 seconds)
python Tests/run_all_tests.py

# Include long regression tests (~5-10 minutes)
python Tests/run_all_tests.py --full

# Verbose mode (see detailed output)
python Tests/run_all_tests.py --verbose
```

Clean pass/fail summary - no scrolling through pages of output!

### 🔧 Sharding Tests (Multi-Device)
```bash
# Test with 1 virtual device
python Tests/run_all_tests.py --sharding 1

# Test with 6 virtual devices
python Tests/run_all_tests.py --sharding 6

# Full sweep: compare 1, 2, 3, 6 devices
python Tests/run_all_tests.py --sharding sweep
```

These run as **separate processes** with proper XLA_FLAGS, so they test
real multi-device behavior (virtual CPUs on laptop, real GPUs on AWS).

### 🧪 Pytest (Alternative)
```bash
# Install pytest first
pip install pytest

# Run all quick tests
pytest Tests/test_solvers.py -v --quick-only

# Run all tests with detailed output
pytest Tests/test_solvers.py -v

# Run only diffusion tests
pytest Tests/test_solvers.py -v -k diffusion

# Run only advection tests
pytest Tests/test_solvers.py -v -k advection

# Minimal dashboard view
pytest Tests/test_solvers.py --tb=no -q
```

**Common Pytest Flags:**

| Flag | Meaning | When to Use |
|------|---------|-------------|
| `-v` | Verbose - show each test name | Debugging: see which tests run |
| `-q` | Quiet - just dots (`.`=pass, `F`=fail, `s`=skip) | Quick dashboard view |
| `--tb=no` | No traceback on failure | Clean summary (hide stack traces) |
| `--tb=short` | Short traceback (1-2 lines) | Debugging failures |
| `--tb=long` | Full traceback | Deep debugging |
| `-k <pattern>` | Run tests matching pattern | `pytest -k diffusion` |
| `-x` | Stop on first failure | Fast debugging |
| `--quick-only` | Skip slow tests (custom marker) | Daily development |

**Example combinations:**
```bash
pytest Tests/ -v --tb=short   # Verbose with readable errors
pytest Tests/ -q --tb=no      # Minimal dashboard
pytest Tests/ -x --tb=short   # Stop on first failure, show error
```

### Quick Verification (fast)
```bash
python Tests/verify_references.py
```
Checks that reference files exist and are loadable.

### Full Validation Suite (slow, ~5-10 minutes)
```bash
# Without pytest
python Tests/test_validation.py

# With pytest (if installed)
pytest Tests/test_validation.py -v -s
```

Runs full regression tests:
- Bitwise reproducibility (exact match to reference)
- Conservation laws (mass, heat)
- Convergence behavior

### Restart Exactness Test (NEW!)
```bash
# Test that stop + restart = continuous run
python Tests/test_restart_exactness.py

# With options:
python Tests/test_restart_exactness.py --steps 100 --restart-at 50 --grid-size 30
```

Validates that:
- Checkpointing preserves full state
- Restarted run produces bitwise-identical results
- No numerical drift from restart

### Sharding Sweep Test (NEW!)
```bash
# Test that 1, 2, 3, 6 devices produce identical results
python Tests/test_sharding_sweep.py

# With options:
python Tests/test_sharding_sweep.py --steps 50 --devices 1,2,3,6 --grid-size 30
```

Validates that:
- Different device counts produce same physics
- Sharding logic doesn't introduce errors
- Code is portable across device configurations

**⚠️ Important: What This Test Actually Tests**

On a laptop (single CPU/GPU), this test validates the sharding **logic** but NOT
actual multi-core parallelism. The XLA_FLAGS trick for virtual devices only works
if set BEFORE Python starts.

For **true multi-device testing**:
- Use multi-GPU hardware (AWS, etc.)
- OR: Run each device count in a separate Python process
- The test confirms results are DETERMINISTIC across configs

### Generating New References
```bash
python Tests/generate_reference_solutions.py
```

**WARNING**: This overwrites existing reference solutions! Only run when you want to update gold standards.

## Test Philosophy

### Tier 1: Unit Tests (future)
Fast tests of individual components:
- Halo exchange correctness
- Limiter behavior
- Coordinate transforms

### Tier 2: Validation Tests (current)
Regression tests against reference solutions:
- Ensure refactoring doesn't break physics
- Detect numerical drift over time
- Verify bitwise reproducibility

### Tier 3: Integration Tests (future)
Multi-device and framework integration:
- Restart exactness (stop/restart = continuous run)
- Sharding reproducibility (1 device = 6 devices)
- I/O correctness (save/load/compare)

## Development Workflow

### Before Refactoring
```bash
# 1. Verify current code passes tests
python Tests/test_validation.py

# 2. Commit passing state
git commit -m "Pre-refactor: all tests passing"
```

### During Refactoring
```bash
# Run tests frequently to catch breaks early
python Tests/test_validation.py
```

### After Refactoring
```bash
# 1. Ensure tests still pass
python Tests/test_validation.py

# 2. If tests fail unexpectedly, debug code (don't update references!)

# 3. Only if physics intentionally changed:
python Tests/generate_reference_solutions.py
git add Tests/validation/*.npz
git commit -m "Update references: improved accuracy via [description]"
```

## Expected Test Results

### Diffusion (Lima Flag)
- **Heat conservation**: Error < 1e-10 (machine precision)
- **Reproducibility**: Matches reference to 1e-12 relative error
- **Physics**: T_max drops from 600K → ~467K over 30 days

### Advection (Cosine Bell)
- **Mass conservation**: Error < 5% for quick test (100 steps)
  - PLR with MC limiter has small numerical diffusion
  - Integrated mass values are O(10¹⁵) because mass = ∫ q × √G × dx²
  - Full 12-day orbit shows ~0.4% error (acceptable for 2nd-order)
- **Reproducibility**: Matches reference to 1e-12 relative error (critical test)
- **Physics**: Peak preserves ~95.6% after full orbit (12 days)

## Future Enhancements

- [ ] Unit tests for individual components
- [x] Sharding reproducibility tests (`test_sharding_sweep.py`)
- [x] Restart exactness tests (`test_restart_exactness.py`)
- [ ] Shallow water equation test cases (Williamson 2, 5, 6; Galewsky)
- [ ] CI/CD integration (GitHub Actions)
- [ ] Performance benchmarking suite

## Notes

- Tests use `config_*_no_sharding.yaml` for deterministic results
- Reference solutions stored as compressed `.npz` (NumPy binary)
- Test suite compatible with pytest but works standalone
- All tests should complete in < 10 minutes on modern hardware

