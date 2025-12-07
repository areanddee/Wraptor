# SWE Development Scratch Area

**Temporary scripts for debugging the Shallow Water Equations solver.**

## Scripts

- `visualize_swe_initial.py` - Generate PNG images of Test Case 2 initial conditions (h, u1, u2)
- `run_swe_short_test.py` - Quick 5-day integration test to check stability

## Usage

```bash
# From Wraptor root:
source /path/to/jax_venv/bin/activate
python Dev/swe_dev/visualize_swe_initial.py
python Dev/swe_dev/run_swe_short_test.py
```

## Note

These are **ephemeral dev scripts** - not part of the test suite. Once the solver is stable, migrate to:
- `Tests/test_swe_testcase2.py` for formal validation
- `Examples/swe_testcase2/` for tutorials

