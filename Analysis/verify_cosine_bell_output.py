"""
Verify Cosine Bell Advection Output - 2D Panel Plots

Creates 2D plots of all 6 faces to verify cosine bell is advecting correctly.

Usage:
    python verify_cosine_bell_output.py /path/to/output_dir
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import zarr
import sys
from pathlib import Path


def load_data(output_dir):
    """Load zarr data."""
    zarr_path = Path(output_dir) / 'output' / 'output.zarr'
    
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
    
    store = zarr.open(str(zarr_path), mode='r')
    
    q_frames = np.array(store['q'])
    times_days = np.array(store['times_days'])
    
    return q_frames, times_days


def plot_6_faces_2d(q_all, time_days, output_path=None):
    """
    Plot all 6 faces in 2D grid.
    
    Args:
        q_all: (6, N, N) concentration field
        time_days: Time in days
        output_path: Where to save (None = show)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Cosine Bell Advection - Day {time_days:.2f}', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    face_names = ['Face 0: +Z (N Pole)', 'Face 1: +Y', 'Face 2: -X',
                  'Face 3: -Y', 'Face 4: +X', 'Face 5: -Z (S Pole)']
    
    # Global color scale
    vmin = 0.0
    vmax = np.max(q_all)
    
    for face in range(6):
        ax = axes[face]
        q_face = q_all[face, :, :]
        
        # Plot (transpose for (j,i) display)
        im = ax.imshow(q_face.T, origin='lower', cmap='viridis',
                      vmin=vmin, vmax=vmax,
                      aspect='equal')
        
        ax.set_title(f'{face_names[face]}\nq_max = {np.max(q_face):.1f}',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('i (ξ¹ direction)', fontsize=9)
        ax.set_ylabel('j (ξ² direction)', fontsize=9)
        
        # Add colorbar to each subplot
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_cosine_bell_output.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    print("="*70)
    print("COSINE BELL ADVECTION VERIFICATION")
    print("="*70)
    print(f"Loading data from: {output_dir}")
    
    # Load data
    q_frames, times_days = load_data(output_dir)
    
    print(f"  Shape: {q_frames.shape}")
    print(f"  Time range: {times_days[0]:.1f} to {times_days[-1]:.1f} days")
    
    # Plot initial frame
    print(f"\nPlotting initial frame (t=0)...")
    q_init = q_frames[0]
    plot_6_faces_2d(q_init, times_days[0], 
                   output_path=f'{output_dir}/verify_frame_0.png')
    
    # Check initial condition
    print(f"\nInitial condition check:")
    print(f"  Max concentration: {np.max(q_init):.1f} (expect ~1000)")
    # Find which face has the bell
    face_maxes = [np.max(q_init[f]) for f in range(6)]
    bell_face = np.argmax(face_maxes)
    print(f"  Bell located on: Face {bell_face}")
    print(f"  Other faces max: {[f'{np.max(q_init[f]):.1f}' for f in range(6)]}")
    
    # Plot final frame
    print(f"\nPlotting final frame (t={times_days[-1]:.1f} days)...")
    q_final = q_frames[-1]
    plot_6_faces_2d(q_final, times_days[-1],
                   output_path=f'{output_dir}/verify_frame_final.png')
    
    # Check advection
    print(f"\nAdvection check (final frame):")
    print(f"  Max concentration: {np.max(q_final):.1f}")
    face_maxes_final = [np.max(q_final[f]) for f in range(6)]
    bell_face_final = np.argmax(face_maxes_final)
    print(f"  Bell located on: Face {bell_face_final}")
    
    if bell_face_final != bell_face:
        print(f"  ✓ Bell moved from Face {bell_face} to Face {bell_face_final}")
    else:
        print(f"  Bell still on Face {bell_face}")
    
    # Conservation check
    N = q_init.shape[1]
    total_init = np.sum(q_init)
    total_final = np.sum(q_final)
    mass_error = (total_final - total_init) / total_init
    print(f"\nMass conservation:")
    print(f"  Initial sum: {total_init:.2e}")
    print(f"  Final sum:   {total_final:.2e}")
    print(f"  Relative error: {mass_error:.2e}")
    
    # Peak reduction (numerical diffusion indicator)
    peak_reduction = (np.max(q_init) - np.max(q_final)) / np.max(q_init) * 100
    print(f"\nNumerical diffusion:")
    print(f"  Peak reduction: {peak_reduction:.1f}%")
    if peak_reduction < 10:
        print(f"  ✓ Good shape preservation (<10%)")
    elif peak_reduction < 30:
        print(f"  ⚠ Moderate diffusion (10-30%)")
    else:
        print(f"  ✗ High diffusion (>30%)")
    
    # Plot middle frame if available
    if len(q_frames) > 2:
        mid_idx = len(q_frames) // 2
        print(f"\nPlotting middle frame (t={times_days[mid_idx]:.1f} days)...")
        q_mid = q_frames[mid_idx]
        plot_6_faces_2d(q_mid, times_days[mid_idx],
                       output_path=f'{output_dir}/verify_frame_mid.png')
    
    print(f"\n{'='*70}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*70}")
    print(f"Check the following files in {output_dir}:")
    print(f"  - verify_frame_0.png (initial bell)")
    print(f"  - verify_frame_mid.png (middle)")
    print(f"  - verify_frame_final.png (after advection)")
    print(f"\nExpected behavior:")
    print(f"  - Bell starts on Face 3 (or 4)")
    print(f"  - Bell advects with solid-body rotation")
    print(f"  - Peak reduces slightly due to numerical diffusion")
    print(f"  - Mass is conserved (error < 1%)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()




