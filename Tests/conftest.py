"""
Pytest Configuration and Shared Fixtures

This file is automatically loaded by pytest and provides:
- Shared fixtures for all tests
- Test configuration
- Custom markers
"""

import pytest
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES: Reference Data
# =============================================================================

@pytest.fixture(scope="session")
def diffusion_reference():
    """Load diffusion reference solution (cached for session)."""
    import numpy as np
    ref_file = Path(__file__).parent / 'validation' / 'diffusion_lima_flag_day30_N120.npz'
    assert ref_file.exists(), f"Missing reference: {ref_file}"
    return np.load(ref_file)


@pytest.fixture(scope="session")
def advection_reference():
    """Load advection reference solution (cached for session)."""
    import numpy as np
    ref_file = Path(__file__).parent / 'validation' / 'advection_cosine_bell_day12_N120.npz'
    assert ref_file.exists(), f"Missing reference: {ref_file}"
    return np.load(ref_file)


# =============================================================================
# FIXTURES: Solvers
# =============================================================================

@pytest.fixture
def diffusion_solver_quick():
    """Create a quick (N=30) diffusion solver for fast tests."""
    from Solvers.fv_cubesphere_diffusion import CubedSphereDiffusion
    config_path = Path(__file__).parent.parent / 'Config' / 'config_diffusion_no_sharding.yaml'
    return CubedSphereDiffusion(N=30, kappa=5e5, config_file=str(config_path))


@pytest.fixture
def advection_solver_quick():
    """Create a quick (N=30) advection solver for fast tests."""
    from Solvers.fv_plr_cubesphere_adv import PLRCubeSphereAdvection
    config_path = Path(__file__).parent.parent / 'Config' / 'config_plr_advection_no_sharding.yaml'
    return PLRCubeSphereAdvection(N=30, config_file=str(config_path))


@pytest.fixture
def diffusion_solver_full(diffusion_reference):
    """Create a full-resolution (N=120) diffusion solver for regression tests."""
    from Solvers.fv_cubesphere_diffusion import CubedSphereDiffusion
    N = int(diffusion_reference['N'])
    kappa = float(diffusion_reference['kappa'])
    config_path = Path(__file__).parent.parent / 'Config' / 'config_diffusion_no_sharding.yaml'
    return CubedSphereDiffusion(N=N, kappa=kappa, config_file=str(config_path))


@pytest.fixture
def advection_solver_full(advection_reference):
    """Create a full-resolution (N=120) advection solver for regression tests."""
    from Solvers.fv_plr_cubesphere_adv import PLRCubeSphereAdvection
    N = int(advection_reference['N'])
    config_path = Path(__file__).parent.parent / 'Config' / 'config_plr_advection_no_sharding.yaml'
    return PLRCubeSphereAdvection(N=N, config_file=str(config_path))


# =============================================================================
# CUSTOM MARKERS
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "quick: mark test as quick (< 1 minute)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (> 1 minute)"
    )
    config.addinivalue_line(
        "markers", "sharding: mark test as requiring multi-device"
    )


# =============================================================================
# PYTEST OPTIONS
# =============================================================================

def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--quick-only",
        action="store_true",
        default=False,
        help="Run only quick tests"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests if --quick-only is specified."""
    if config.getoption("--quick-only"):
        skip_slow = pytest.mark.skip(reason="--quick-only specified")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

