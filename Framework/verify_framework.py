"""
Framework Verification Script

Quick test to verify Framework modules load correctly.
Run this after creating/modifying Framework to catch import errors early.
"""

import sys
from pathlib import Path

# Add JaxStream2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all Framework modules import correctly."""
    print("="*70)
    print("FRAMEWORK VERIFICATION")
    print("="*70)
    
    errors = []
    
    # Test solver_interface
    print("\n1. Testing solver_interface.py...")
    try:
        from Framework.solver_interface import (
            NumericalSolver, OutputSpec, SolverState,
            validate_solver_interface
        )
        print("  ✓ solver_interface imports successfully")
        print(f"    - NumericalSolver: {NumericalSolver}")
        print(f"    - OutputSpec: {OutputSpec}")
    except Exception as e:
        print(f"  ✗ solver_interface import failed: {e}")
        errors.append(('solver_interface', e))
    
    # Test io_manager
    print("\n2. Testing io_manager.py...")
    try:
        from Framework.io_manager import IOManager, validate_output_config
        print("  ✓ io_manager imports successfully")
        print(f"    - IOManager: {IOManager}")
    except Exception as e:
        print(f"  ✗ io_manager import failed: {e}")
        errors.append(('io_manager', e))
    
    # Test runner
    print("\n3. Testing runner.py...")
    try:
        from Framework.runner import run_simulation, load_solver
        print("  ✓ runner imports successfully")
        print(f"    - run_simulation: {run_simulation}")
        print(f"    - load_solver: {load_solver}")
    except Exception as e:
        print(f"  ✗ runner import failed: {e}")
        errors.append(('runner', e))
    
    # Summary
    print("\n" + "="*70)
    if errors:
        print(f"❌ VERIFICATION FAILED ({len(errors)} errors)")
        print("="*70)
        for module, error in errors:
            print(f"\n{module}:")
            print(f"  {error}")
        return False
    else:
        print("✅ ALL FRAMEWORK MODULES VERIFIED")
        print("="*70)
        print("\nFramework structure:")
        print("  Framework/")
        print("  ├── solver_interface.py  ✓ (Abstract base class)")
        print("  ├── io_manager.py        ✓ (Zarr + Orbax I/O)")
        print("  └── runner.py            ✓ (Main execution driver)")
        print("\nReady for solver refactoring!")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

