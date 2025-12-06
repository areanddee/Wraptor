"""
Verify Lima Flag Output - Simple 2D Diagnostic Plots

This script loads Zarr output and creates simple 2D plots of all 6 faces
to verify the data is correct (independent of 3D sphere visualization).

Usage:
    python verify_lima_flag_output.py lima_test_5days
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import zarr
import sys
from pathlib import Path

def plot_6_faces_2d(T_all, time_days, output_path='verify_output.png'):
    """
    Plot all 6 faces in simple 2D grid (like original test_temperature_diffusion.py).
    
    Args:
        T_all: (6, N, N) temperature array
        time_days: Time in days
        output_path: Where to save
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    face_names = ['Face 0: +Z (N Pole)', 'Face 1: +Y', 'Face 2: -X',
                  'Face 3: -Y', 'Face 4: +X', 'Face 5: -Z (S Pole)']
    
    # Global color scale - logarithmic 1K to 1000K
    vmin = 1.0
    vmax = 1000.0
    
    for face in range(6):
        ax = axes[face]
        T_face = T_all[face, :, :]
        
        # Plot with log scale (1K to 1000K)
        T_plot = np.clip(T_face.T, vmin, vmax)  # Transpose for (j,i) display, clip to range
        im = ax.imshow(T_plot, origin='lower', cmap='magma',
                      norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                      aspect='equal')
        
        ax.set_title(f'{face_names[face]}\nT_max = {np.max(T_face):.1f}K',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('i (ξ¹ direction)', fontsize=9)
        ax.set_ylabel('j (ξ² direction)', fontsize=9)
        
        # Add max temp annotation
        ax.text(0.95, 0.05, f'{np.max(T_face):.1f}°C',
               transform=ax.transAxes, fontsize=8, 
               ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Lima Flag Temperature at t = {time_days:.1f} days\n'
                f'(2D Face View - Verify Data Correctness)',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def verify_lima_flag_data(output_dir):
    """
    Load Zarr data and create diagnostic plots.
    
    Args:
        output_dir: Directory containing output/output.zarr
    """
    print("="*70)
    print("LIMA FLAG DATA VERIFICATION")
    print("="*70)
    
    # Load data
    zarr_path = Path(output_dir) / 'output' / 'output.zarr'
    if not zarr_path.exists():
        print(f"Error: {zarr_path} not found")
        sys.exit(1)
    
    store = zarr.open(str(zarr_path), mode='r')
    T_frames = np.array(store['T'][:])
    
    if 'metadata' in store:
        metadata_store = zarr.open(str(zarr_path / 'metadata'), mode='r')
        metadata = np.array(metadata_store[:])
        times = metadata[:, 0]
    else:
        times = np.arange(T_frames.shape[0]) * 86400.0
    
    times_days = times / 86400.0
    
    print(f"\nData loaded:")
    print(f"  Frames: {T_frames.shape[0]}")
    print(f"  Shape: {T_frames.shape}")
    print(f"  Time range: {times_days[0]:.1f} to {times_days[-1]:.1f} days")
    
    # Plot initial frame
    print(f"\nPlotting initial frame (t=0)...")
    T_init = T_frames[0]
    plot_6_faces_2d(T_init, times_days[0], 
                   output_path=f'{output_dir}/verify_frame_0.png')
    
    # Check initial condition
    print(f"\nInitial condition check:")
    print(f"  Face 0 upper-left (i<N/2, j<N/2): T_max = {np.max(T_init[0, :15, :15]):.1f}K (expect 600K)")
    print(f"  Face 0 lower-right (i>=N/2, j>=N/2): T_max = {np.max(T_init[0, 15:, 15:]):.1f}K (expect 600K)")
    print(f"  Face 0 upper-right (i<N/2, j>=N/2): T_max = {np.max(T_init[0, :15, 15:]):.1f}K (expect 1K background)")
    print(f"  Face 0 lower-left (i>=N/2, j<N/2): T_max = {np.max(T_init[0, 15:, :15]):.1f}K (expect 1K background)")
    print(f"  Face 1 (background): T_max = {np.max(T_init[1]):.1f}K (expect 1K)")
    print(f"  Face 2 (background): T_max = {np.max(T_init[2]):.1f}K (expect 1K)")
    
    # Plot final frame
    print(f"\nPlotting final frame (t={times_days[-1]:.1f} days)...")
    T_final = T_frames[-1]
    plot_6_faces_2d(T_final, times_days[-1],
                   output_path=f'{output_dir}/verify_frame_final.png')
    
    # Check diffusion
    print(f"\nDiffusion check (final frame):")
    print(f"  Face 0 T_max: {np.max(T_final[0]):.1f}K (hottest - source)")
    print(f"  Face 1 T_max: {np.max(T_final[1]):.1f}K (should have heat from Face 0)")
    print(f"  Face 2 T_max: {np.max(T_final[2]):.1f}K (should have heat from Face 0)")
    print(f"  Face 3 T_max: {np.max(T_final[3]):.1f}K (should have heat from Face 0)")
    print(f"  Face 4 T_max: {np.max(T_final[4]):.1f}K (should have heat from Face 0)")
    print(f"  Face 5 T_max: {np.max(T_final[5]):.1f}K (coldest - far from source, near 1K background)")
    
    # Plot middle frame if available
    if len(T_frames) > 2:
        mid_idx = len(T_frames) // 2
        print(f"\nPlotting middle frame (t={times_days[mid_idx]:.1f} days)...")
        T_mid = T_frames[mid_idx]
        plot_6_faces_2d(T_mid, times_days[mid_idx],
                       output_path=f'{output_dir}/verify_frame_mid.png')
    
    print(f"\n{'='*70}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*70}")
    print(f"Check the following files in {output_dir}:")
    print(f"  - verify_frame_0.png (initial Lima Flag)")
    print(f"  - verify_frame_mid.png (middle)")
    print(f"  - verify_frame_final.png (final)")
    print(f"\nIf these look correct but 3D sphere doesn't:")
    print(f"  → Problem is in cubesphere_diffusion_viz.py geometry mapping")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_lima_flag_output.py <output_directory>")
        print("Example: python verify_lima_flag_output.py lima_test_5days")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    verify_lima_flag_data(output_dir)

