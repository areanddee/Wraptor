# Development Directory

This directory contains **temporary development and debugging scripts** that are not part of the production codebase.

## Structure

```
Dev/
├── README.md              # This file
├── swe_design.md          # Design doc for SWE implementation (commit this)
├── swe_dev/               # SWE development scratch area (ephemeral)
│   ├── visualize_swe_initial.py
│   └── run_swe_short_test.py
└── [future_feature_dev]/  # Create subdirs for new feature development
```

## Guidelines

### What belongs in `Dev/`?

- **Design documents** (`.md` files) - commit these
- **Exploratory scripts** for debugging new features - DON'T commit
- **Quick tests** to verify behavior during development - DON'T commit
- **Prototypes** before formal implementation - review before committing

### What DOESN'T belong in `Dev/`?

- **Production tests** → Use `Tests/` with proper pytest structure
- **Examples/tutorials** → Use `Examples/`
- **Production drivers** → Use `Framework/` or create solver-specific scripts

### Output files

All output from dev scripts (`.png`, `.zarr/`, `output*/`) is `.gitignore`'d automatically.

## Workflow

1. Create a subdirectory for your feature (e.g., `Dev/swe_dev/`)
2. Hack away with quick scripts
3. When stable, migrate to:
   - Formal tests → `Tests/test_*.py`
   - Examples → `Examples/`
   - Documentation → `README.md` or `Dev/*.md`
4. Delete or archive the dev scripts

## Current Active Development

- **`swe_dev/`**: Shallow Water Equations Test Case 2 implementation and validation

