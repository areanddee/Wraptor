"""
Simple GPU test for diffusion solver.
Run this before the full test suite to verify GPU is working.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp


def test_jax_gpu():
    """Verify JAX can see and use GPU."""
    print("\n" + "="*70)
    print("JAX GPU DETECTION TEST")
    print("="*70)
    
    devices = jax.devices()
    backend = jax.default_backend()
    
    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {backend}")
    print(f"Devices: {devices}")
    print(f"Number of devices: {len(devices)}")
    
    if backend == 'cpu':
        print("\n⚠️  WARNING: JAX is using CPU, not GPU!")
        print("Set: export JAX_PLATFORMS=cuda")
        return False
    
    print(f"\n✅ JAX is using GPU backend: {backend}")
    return True


def test_simple_computation():
    """Test simple GPU computation."""
    print("\n" + "="*70)
    print("SIMPLE GPU COMPUTATION TEST")
    print("="*70)
    
    # Create arrays on GPU
    N = 1000
    x = jnp.ones((N, N))
    y = jnp.ones((N, N))
    
    # JIT compile
    @jax.jit
    def matmul(a, b):
        return jnp.dot(a, b)
    
    # Warmup
    print("Warming up JIT...")
    _ = matmul(x, y).block_until_ready()
    
    # Timed run
    print("Running timed matrix multiply...")
    start = time.time()
    result = matmul(x, y).block_until_ready()
    elapsed = time.time() - start
    
    print(f"Matrix multiply ({N}x{N}): {elapsed*1000:.2f} ms")
    print(f"Result shape: {result.shape}")
    print(f"Result sum: {jnp.sum(result):.2f} (expected {N*N*N:.2f})")
    
    # Check correctness
    expected = N * N * N
    actual = float(jnp.sum(result))
    error = abs(actual - expected) / expected
    
    if error < 1e-5:
        print(f"\n✅ Computation correct (error: {error:.2e})")
        return True
    else:
        print(f"\n❌ Computation error: {error:.2e}")
        return False


def test_diffusion_solver():
    """Test actual diffusion solver on GPU."""
    print("\n" + "="*70)
    print("DIFFUSION SOLVER TEST")
    print("="*70)
    
    try:
        from Solvers.fv_cubesphere_diffusion import CubedSphereDiffusion
        import yaml
        
        # Load minimal config
        config_path = Path(__file__).parent.parent / "Config" / "diffusion_framework.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        config['solver']['N'] = 30  # Small for quick test
        config['parallelization']['enable_sharding'] = False  # Single GPU, no sharding
        
        print(f"Initializing solver (N={config['solver']['N']})...")
        solver = CubedSphereDiffusion(
            N=config['solver']['N'],
            kappa=config['solver']['kappa'],
            config_file=None
        )
        
        # Initialize
        print("Initializing state...")
        state = solver.initialize({'pattern': 'lima_flag', 'T_hot': 600.0})
        
        # Take 5 steps
        print("Running 5 timesteps...")
        dt = 100.0  # seconds
        for i in range(5):
            state = solver.step(state, dt)
            print(f"  Step {i+1}: T_max={jnp.max(state.T):.2f} K")
        
        # Check result
        T_max = float(jnp.max(state.T))
        if 200 < T_max < 700:
            print(f"\n✅ Diffusion solver working (T_max={T_max:.2f} K)")
            return True
        else:
            print(f"\n❌ Suspicious T_max: {T_max:.2f} K")
            return False
            
    except Exception as e:
        print(f"\n❌ Diffusion solver failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all GPU tests."""
    print("\n" + "="*70)
    print("WRAPTOR GPU TEST SUITE")
    print("="*70)
    
    results = {
        'JAX GPU Detection': test_jax_gpu(),
        'Simple Computation': test_simple_computation(),
        'Diffusion Solver': test_diffusion_solver()
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print("="*70)
    
    # Exit code
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All GPU tests PASSED!")
        return 0
    else:
        print("\n❌ Some GPU tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())



