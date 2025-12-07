"""
Quick script to initialize SWE Test Case 2 and visualize initial state.
"""

import sys
from pathlib import Path
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Framework.runner import load_solver
from Solvers.geometry import CubedSphereGeometry


def plot_field_on_sphere(field_data, title, output_path, vmin=None, vmax=None, cmap='viridis'):
    """
    Plot a field on the cubed sphere using lat-lon interpolation.
    
    Args:
        field_data: (6, N, N) array of field values on cubed sphere
        title: Plot title
        output_path: Where to save the PNG
        vmin, vmax: Color scale limits (auto if None)
        cmap: Colormap name
    """
    from scipy.interpolate import griddata
    
    num_faces, N, _ = field_data.shape
    geometry = CubedSphereGeometry.create(N)
    
    # Get coordinates for all faces (returns tuple of x, y, z arrays)
    x_coords, y_coords, z_coords = geometry.get_xyz_all_faces()
    
    # Convert to lat-lon
    x_flat = x_coords.ravel()
    y_flat = y_coords.ravel()
    z_flat = z_coords.ravel()
    
    lon = np.arctan2(y_flat, x_flat)
    lat = np.arcsin(np.clip(z_flat, -1, 1))
    
    field_flat = field_data.ravel()
    
    # Create lat-lon grid
    lon_grid = np.linspace(-np.pi, np.pi, 360)
    lat_grid = np.linspace(-np.pi/2, np.pi/2, 180)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Interpolate
    field_interp = griddata(
        (lon, lat),
        field_flat,
        (lon_mesh, lat_mesh),
        method='linear'
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if vmin is None:
        vmin = np.nanmin(field_interp)
    if vmax is None:
        vmax = np.nanmax(field_interp)
    
    im = ax.contourf(
        np.degrees(lon_mesh),
        np.degrees(lat_mesh),
        field_interp,
        levels=50,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    cbar = fig.colorbar(im, ax=ax)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_path}")


def main():
    # Load config
    config_path = Path(__file__).parent.parent.parent / "Config" / "swe_framework.yaml"
    print(f"Loading config: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Add config file path for reference
    config['_config_file_path'] = str(config_path)
    
    # Load solver
    print("\nInitializing solver...")
    solver = load_solver(config)
    
    # Initialize state
    print("\nGenerating initial conditions (Test Case 2)...")
    state = solver.initialize(config)
    
    # Print statistics
    h_arr = np.array(state.h)
    hu1_arr = np.array(state.hu1)
    hu2_arr = np.array(state.hu2)
    
    # Compute velocities from momentum
    u1 = hu1_arr / h_arr
    u2 = hu2_arr / h_arr
    
    print(f"\nInitial state statistics:")
    print(f"  h:  min={h_arr.min():.2f} m, max={h_arr.max():.2f} m, mean={h_arr.mean():.2f} m")
    print(f"  u1: min={u1.min():.4f} m/s, max={u1.max():.4f} m/s")
    print(f"  u2: min={u2.min():.4f} m/s, max={u2.max():.4f} m/s")
    
    # Create output directory
    output_dir = Path(__file__).parent / "output_swe_initial"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    
    # Plot h (depth)
    plot_field_on_sphere(
        h_arr,
        "SWE Test Case 2: Initial Depth Field (h)",
        output_dir / "initial_h.png",
        cmap='Blues'
    )
    
    # Plot u1 (contravariant velocity component 1)
    plot_field_on_sphere(
        u1,
        "SWE Test Case 2: Initial u1 velocity",
        output_dir / "initial_u1.png",
        cmap='RdBu_r'
    )
    
    # Plot u2 (contravariant velocity component 2)
    plot_field_on_sphere(
        u2,
        "SWE Test Case 2: Initial u2 velocity",
        output_dir / "initial_u2.png",
        cmap='RdBu_r'
    )
    
    print(f"\n✅ Initial conditions visualized in: {output_dir}")


if __name__ == "__main__":
    main()

