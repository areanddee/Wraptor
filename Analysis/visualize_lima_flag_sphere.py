"""
Lima Flag Sphere Visualization - Using Our Coordinate Transform

This script computes lat-lon directly from our equiangular_to_xyz_face function
(which we know is correct from the 2D verification), then plots on sphere.

No dependency on external geometry files - uses our own proven coordinate mapping!

Usage:
    python visualize_lima_flag_sphere.py lima_test_5days --frame 0
    python visualize_lima_flag_sphere.py lima_test_5days --movie --fps 10
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import zarr
import sys
from pathlib import Path
import argparse

# Import our coordinate transform
from fv_cubesphere_diffusion import equiangular_to_xyz_face

R_SPHERE = 6.371e6  # meters


def compute_latlon_from_our_coordinates(N):
    """
    Compute lat-lon using OUR coordinate transform (the one we know works!).
    
    Returns:
        lon_all, lat_all: (6, N, N) arrays of lon, lat in radians
        X_all, Y_all, Z_all: (6, N, N) arrays of Cartesian coords
    """
    # Create grid
    xi1_1d = np.linspace(-np.pi/4, np.pi/4, N)
    xi2_1d = np.linspace(-np.pi/4, np.pi/4, N)
    XI1, XI2 = np.meshgrid(xi1_1d, xi2_1d, indexing='ij')
    
    lon_all = np.zeros((6, N, N))
    lat_all = np.zeros((6, N, N))
    X_all = np.zeros((6, N, N))
    Y_all = np.zeros((6, N, N))
    Z_all = np.zeros((6, N, N))
    
    for face in range(6):
        # Get Cartesian coords from our transform
        X, Y, Z = equiangular_to_xyz_face(XI1, XI2, face, R=R_SPHERE)
        X_all[face] = np.array(X)
        Y_all[face] = np.array(Y)
        Z_all[face] = np.array(Z)
        
        # Compute lat-lon from Cartesian
        lon_all[face] = np.arctan2(Y, X)
        lat_all[face] = np.arcsin(Z / R_SPHERE)
    
    return lon_all, lat_all, X_all, Y_all, Z_all


def interpolate_to_sphere(T_cubesphere, lon_all, lat_all, nlat=360, nlon=720):
    """
    Interpolate cubed-sphere data to regular lat-lon grid.
    
    Args:
        T_cubesphere: (6, N, N) temperature on cubed-sphere
        lon_all, lat_all: (6, N, N) coordinates from our transform
        nlat, nlon: Target resolution
        
    Returns:
        lon_grid, lat_grid: (nlat, nlon) regular grids
        T_interp: (nlat, nlon) interpolated temperature
    """
    # Flatten cubed-sphere data
    lon_flat = lon_all.flatten()
    lat_flat = lat_all.flatten()
    T_flat = T_cubesphere.flatten()
    
    # Create regular lat-lon grid
    lat_1d = np.linspace(-np.pi/2, np.pi/2, nlat)
    lon_1d = np.linspace(-np.pi, np.pi, nlon)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
    
    # Interpolate (LINEAR - preserves physical bounds, no overshoot!)
    points = np.column_stack([lon_flat, lat_flat])
    T_interp = griddata(points, T_flat, (lon_grid, lat_grid), method='linear')
    
    # Fill any NaN values (e.g., at poles)
    if np.any(np.isnan(T_interp)):
        mask = np.isnan(T_interp)
        T_interp[mask] = griddata(points, T_flat, (lon_grid[mask], lat_grid[mask]), method='nearest')
    
    return lon_grid, lat_grid, T_interp


def plot_sphere(lon_grid, lat_grid, T_interp, time_days,
                elev=30, azim=45, cmap='magma', 
                output_path=None, dpi=150):
    """
    Plot temperature on 3D sphere.
    
    Args:
        lon_grid, lat_grid: (nlat, nlon) coordinate grids
        T_interp: (nlat, nlon) interpolated temperature
        time_days: Time in days
        elev, azim: Viewing angles
        cmap: Colormap
        output_path: Where to save (None = show)
        dpi: Resolution
    """
    # Convert to Cartesian for 3D plotting
    X_sphere = np.cos(lat_grid) * np.cos(lon_grid)
    Y_sphere = np.cos(lat_grid) * np.sin(lon_grid)
    Z_sphere = np.sin(lat_grid)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color normalization - LOGARITHMIC scale (1K to 1000K)
    vmin = 1.0
    vmax = 1000.0
    
    # Logarithmic normalization (10^0 to 10^3)
    norm = plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    colors_normalized = norm(np.clip(T_interp, vmin, vmax))
    
    # Plot surface (use all interpolated points - no stride!)
    surf = ax.plot_surface(X_sphere, Y_sphere, Z_sphere,
                          facecolors=plt.colormaps[cmap](colors_normalized),
                          rstride=1, cstride=1,
                          shade=False,
                          antialiased=True)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('Temperature (K)', fontsize=12)
    
    # Setup axes
    limit = 1.1
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_box_aspect([1, 1, 1])
    
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_zlabel('Z', fontsize=11)
    
    # Title
    ax.set_title(f'Lima Flag Temperature Diffusion - Day {time_days:.1f}',
                fontsize=13, pad=15, fontweight='bold')
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Clean up axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    else:
        plt.show()


def generate_movie(T_frames, times_days, N, output_path='lima_flag_sphere.mp4',
                  fps=15, dpi=150, elev=30, azim=45, nlat=180, nlon=360, skip_first=False):
    """
    Generate movie from all frames.
    
    Args:
        skip_first: If True, skip frame 0 (initial condition)
    """
    try:
        from matplotlib.animation import FuncAnimation, FFMpegWriter
    except ImportError:
        print("Error: matplotlib.animation not available")
        print("Install ffmpeg: brew install ffmpeg")
        return None
    
    # Skip first frame if requested
    if skip_first and len(T_frames) > 1:
        T_frames = T_frames[1:]
        times_days = times_days[1:]
        print(f"⚠ Skipping frame 0 (initial condition)")
    
    n_frames = len(T_frames)
    print(f"\n{'='*70}")
    print("GENERATING MOVIE")
    print(f"{'='*70}")
    print(f"Frames: {n_frames}")
    print(f"Time range: {times_days[0]:.1f} to {times_days[-1]:.1f} days")
    print(f"FPS: {fps}")
    print(f"Duration: {n_frames/fps:.1f} seconds")
    
    # Compute coordinates once
    print("Computing lat-lon coordinates from our transform...")
    lon_all, lat_all, _, _, _ = compute_latlon_from_our_coordinates(N)
    
    # Pre-compute regular grid
    lat_1d = np.linspace(-np.pi/2, np.pi/2, nlat)
    lon_1d = np.linspace(-np.pi, np.pi, nlon)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
    
    # Cartesian for plotting
    X_sphere = np.cos(lat_grid) * np.cos(lon_grid)
    Y_sphere = np.cos(lat_grid) * np.sin(lon_grid)
    Z_sphere = np.sin(lat_grid)
    
    # Color scale - LOGARITHMIC (1K to 1000K)
    vmin = 1.0
    vmax = 1000.0
    norm = plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    
    print(f"Temperature range: [{vmin:.1f}, {vmax:.1f}] K (logarithmic scale: 10^0 to 10^3)")
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Setup axes (static)
    limit = 1.1
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_zlabel('Z', fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.2)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='magma', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('Temperature (K)', fontsize=12)
    
    # Animation update
    surf = [None]
    
    def update(frame_idx):
        # Progress bar
        percent = 100 * (frame_idx + 1) / n_frames
        bar_length = 50
        filled = int(bar_length * (frame_idx + 1) / n_frames)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r  Progress: [{bar}] {percent:.1f}% ({frame_idx+1}/{n_frames})", end='', flush=True)
        
        # Clear previous
        if surf[0] is not None:
            surf[0].remove()
        
        # Interpolate this frame
        _, _, T_interp = interpolate_to_sphere(T_frames[frame_idx], lon_all, lat_all, nlat, nlon)
        
        # Plot (logarithmic normalization, clipped to 1-1000K)
        colors_normalized = norm(np.clip(T_interp, vmin, vmax))
        surf[0] = ax.plot_surface(X_sphere, Y_sphere, Z_sphere,
                                  facecolors=plt.colormaps['magma'](colors_normalized),
                                  rstride=1, cstride=1,
                                  shade=False,
                                  antialiased=True)
        
        # Update title
        ax.set_title(f'Lima Flag Temperature Diffusion - Day {times_days[frame_idx]:.1f}',
                    fontsize=13, pad=15, fontweight='bold')
        
        return [surf[0]]
    
    # Create animation
    print("  Rendering frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    
    # Save
    print()  # Newline after progress bar
    print(f"  Writing to {output_path}...")
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    anim.save(output_path, writer=writer, dpi=dpi)
    
    plt.close()
    
    print(f"  ✓ Movie saved: {output_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Lima Flag Sphere Visualization (using our coordinate transform)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output_dir', type=str,
                        help='Output directory containing output/output.zarr')
    parser.add_argument('--frame', type=int,
                        help='Plot single frame index (-1 = last)')
    parser.add_argument('--movie', action='store_true',
                        help='Generate movie from all frames')
    parser.add_argument('--skip-first-frame', action='store_true',
                        help='Skip frame 0 (initial condition) in movie')
    parser.add_argument('--output', type=str,
                        help='Output file path')
    parser.add_argument('--elev', type=float, default=30,
                        help='Elevation angle (degrees)')
    parser.add_argument('--azim', type=float, default=45,
                        help='Azimuth angle (degrees)')
    parser.add_argument('--fps', type=int, default=15,
                        help='Movie FPS')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Resolution (DPI)')
    parser.add_argument('--nlat', type=int, default=360,
                        help='Interpolation latitude resolution (default: 360)')
    parser.add_argument('--nlon', type=int, default=720,
                        help='Interpolation longitude resolution (default: 720)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LIMA FLAG SPHERE VISUALIZATION")
    print("="*70)
    print("Using OUR coordinate transform (equiangular_to_xyz_face)")
    print("="*70)
    
    # Load data
    zarr_path = Path(args.output_dir) / 'output' / 'output.zarr'
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
    N = T_frames.shape[2]
    
    print(f"\nData loaded:")
    print(f"  Frames: {T_frames.shape[0]}")
    print(f"  Grid: {N}×{N} per face")
    print(f"  Time range: {times_days[0]:.1f} to {times_days[-1]:.1f} days")
    
    # Compute coordinates from OUR transform
    print(f"\nComputing lat-lon from our coordinate transform...")
    lon_all, lat_all, _, _, _ = compute_latlon_from_our_coordinates(N)
    print(f"  ✓ Coordinates computed for all 6 faces")
    
    # Single frame
    if args.frame is not None:
        frame_idx = args.frame if args.frame >= 0 else len(T_frames) + args.frame
        
        print(f"\nPlotting frame {frame_idx} (t={times_days[frame_idx]:.1f} days)...")
        print(f"  Interpolating to {args.nlat}×{args.nlon} lat-lon grid...")
        
        lon_grid, lat_grid, T_interp = interpolate_to_sphere(
            T_frames[frame_idx], lon_all, lat_all, args.nlat, args.nlon
        )
        
        output = args.output or f'{args.output_dir}/sphere_frame_{frame_idx}.png'
        plot_sphere(lon_grid, lat_grid, T_interp, times_days[frame_idx],
                   elev=args.elev, azim=args.azim,
                   output_path=output, dpi=args.dpi)
        
        print(f"\n✓ Done! Check: {output}")
    
    # Movie
    elif args.movie:
        output = args.output or f'{args.output_dir}/lima_flag_sphere_movie.mp4'
        generate_movie(T_frames, times_days, N, output_path=output,
                      fps=args.fps, dpi=args.dpi,
                      elev=args.elev, azim=args.azim,
                      nlat=args.nlat, nlon=args.nlon,
                      skip_first=args.skip_first_frame)
        print(f"\n✓ Done! Movie: {output}")
    
    else:
        parser.print_help()
        print("\nError: Must specify --frame or --movie")
        sys.exit(1)


if __name__ == "__main__":
    main()

