"""
Quick verification script to check reference solutions are valid
Run this before full validation tests to catch any issues early
"""

import numpy as np
from pathlib import Path

def verify_diffusion_reference():
    """Verify diffusion reference file"""
    print("\nDiffusion Reference:")
    print("-" * 50)
    
    ref_file = Path(__file__).parent / 'validation' / 'diffusion_lima_flag_day30_N120.npz'
    
    if not ref_file.exists():
        print(f"  ✗ MISSING: {ref_file}")
        return False
    
    data = np.load(ref_file)
    print(f"  ✓ File exists: {ref_file.name}")
    print(f"  ✓ Size: {ref_file.stat().st_size / 1024:.1f} KB")
    print(f"  ✓ Keys: {list(data.keys())}")
    print(f"  ✓ T shape: {data['T'].shape}")
    print(f"  ✓ T_max: {np.max(data['T']):.2f} K")
    print(f"  ✓ T_min: {np.min(data['T']):.2f} K")
    print(f"  ✓ N: {data['N']}")
    print(f"  ✓ days: {data['days']}")
    
    return True


def verify_advection_reference():
    """Verify advection reference file"""
    print("\nAdvection Reference:")
    print("-" * 50)
    
    ref_file = Path(__file__).parent / 'validation' / 'advection_cosine_bell_day12_N120.npz'
    
    if not ref_file.exists():
        print(f"  ✗ MISSING: {ref_file}")
        return False
    
    data = np.load(ref_file)
    print(f"  ✓ File exists: {ref_file.name}")
    print(f"  ✓ Size: {ref_file.stat().st_size / 1024:.1f} KB")
    print(f"  ✓ Keys: {list(data.keys())}")
    print(f"  ✓ q shape: {data['q'].shape}")
    print(f"  ✓ q_max: {np.max(data['q']):.2f}")
    print(f"  ✓ q_min: {np.min(data['q']):.2f}")
    print(f"  ✓ N: {data['N']}")
    print(f"  ✓ days: {data['days']}")
    
    return True


if __name__ == "__main__":
    print("="*70)
    print("REFERENCE SOLUTION VERIFICATION")
    print("="*70)
    
    diff_ok = verify_diffusion_reference()
    adv_ok = verify_advection_reference()
    
    print("\n" + "="*70)
    if diff_ok and adv_ok:
        print("✅ ALL REFERENCE FILES VALID!")
        print("="*70)
        print("\nReady to run validation tests:")
        print("  python Tests/test_validation.py")
    else:
        print("❌ MISSING REFERENCE FILES!")
        print("="*70)
        print("\nGenerate them first:")
        print("  python Tests/generate_reference_solutions.py")
    print("="*70 + "\n")

