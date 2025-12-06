"""
2D Periodic Finite Volume Shallow Water Solver

Example/demo solver showing Framework integration.
Based on Rusanov flux method for periodic domain.

This serves as a reference implementation for new solvers.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from typing import Dict, Any, List, Tuple
import sys
from pathlib import Path

# Add Framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Framework.solver_interface import NumericalSolver, OutputSpec


# ============================================================================
# STATE
# ============================================================================

@struct.dataclass
class FVPeriodicState:
    """State for 2D periodic FV solver"""
    h: jax.Array  # (nx, ny) - height field
    u: jax.Array  # (nx, ny) - x-velocity
    v: jax.Array  # (nx, ny) - y-velocity
    time: float
    step: int


# ============================================================================
# SOLVER
# ============================================================================

class FVPeriodic2D(NumericalSolver):
    """
    2D Periodic Finite Volume Shallow Water
    
    Domain: Periodic torus
    Method: Rusanov (local Lax-Friedrichs) flux
    Time integration: RK2
    
    This is a complete, working example of Framework integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize solver from config.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.g = config['physics']['gravity']
        self.f = config['physics'].get('coriolis', 0.0)
        self.nx = config['grid']['nx']
        self.ny = config['grid']['ny']
        self.Lx = float(config['domain']['Lx'])
        self.Ly = float(config['domain']['Ly'])
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        
        print(f"\nFVPeriodic2D Solver:")
        print(f"  Grid: {self.nx} × {self.ny}")
        print(f"  Domain: {self.Lx} × {self.Ly} m")
        print(f"  Resolution: dx={self.dx:.1f} m, dy={self.dy:.1f} m")
        print(f"  Physics: g={self.g} m/s², f={self.f} rad/s")
    
    def initialize(self, config: Dict[str, Any]) -> FVPeriodicState:
        """Initialize geostrophic jet with perturbation."""
        nx, ny = self.nx, self.ny
        Lx, Ly = self.Lx, self.Ly
        
        x = jnp.linspace(0, Lx, nx, endpoint=False)
        y = jnp.linspace(0, Ly, ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        
        # Geostrophic jet
        y_center = Ly / 2
        width = Ly / 10
        u_max = config.get('ic_params', {}).get('u_max', 20.0)
        
        u = u_max * jnp.exp(-((Y - y_center) / width)**2)
        v = jnp.zeros_like(u)
        
        # Geostrophic height
        h_mean = config.get('ic_params', {}).get('h_mean', 1000.0)
        h_pert = -(self.f * u_max * width / self.g) * jnp.sqrt(jnp.pi) * \
                 jax.scipy.special.erf((Y - y_center) / width)
        h = h_mean + h_pert
        
        # Perturbation
        pert_amp = config.get('ic_params', {}).get('perturbation', 10.0)
        perturbation = pert_amp * jnp.sin(2 * jnp.pi * X / Lx) * \
                                  jnp.sin(2 * jnp.pi * Y / Ly)
        h = h + perturbation
        
        return FVPeriodicState(h=h, u=u, v=v, time=0.0, step=0)
    
    @staticmethod
    @jax.jit
    def _rusanov_flux_x(h_L, h_R, u_L, u_R, v_L, v_R, g):
        """Rusanov flux in x-direction"""
        # Conserved variables
        q1_L, q2_L, q3_L = h_L, h_L * u_L, h_L * v_L
        q1_R, q2_R, q3_R = h_R, h_R * u_R, h_R * v_R
        
        # Physical fluxes
        f1_L = h_L * u_L
        f2_L = h_L * u_L * u_L + 0.5 * g * h_L * h_L
        f3_L = h_L * u_L * v_L
        
        f1_R = h_R * u_R
        f2_R = h_R * u_R * u_R + 0.5 * g * h_R * h_R
        f3_R = h_R * u_R * v_R
        
        # Wave speeds
        c_L = jnp.sqrt(g * jnp.maximum(h_L, 1e-10))
        c_R = jnp.sqrt(g * jnp.maximum(h_R, 1e-10))
        alpha = jnp.maximum(jnp.abs(u_L) + c_L, jnp.abs(u_R) + c_R)
        
        # Rusanov flux
        flux_h  = 0.5 * (f1_L + f1_R - alpha * (q1_R - q1_L))
        flux_hu = 0.5 * (f2_L + f2_R - alpha * (q2_R - q2_L))
        flux_hv = 0.5 * (f3_L + f3_R - alpha * (q3_R - q3_L))
        
        return flux_h, flux_hu, flux_hv
    
    @staticmethod
    @jax.jit
    def _rusanov_flux_y(h_L, h_R, u_L, u_R, v_L, v_R, g):
        """Rusanov flux in y-direction"""
        q1_L, q2_L, q3_L = h_L, h_L * u_L, h_L * v_L
        q1_R, q2_R, q3_R = h_R, h_R * u_R, h_R * v_R
        
        g1_L = h_L * v_L
        g2_L = h_L * u_L * v_L
        g3_L = h_L * v_L * v_L + 0.5 * g * h_L * h_L
        
        g1_R = h_R * v_R
        g2_R = h_R * u_R * v_R
        g3_R = h_R * v_R * v_R + 0.5 * g * h_R * h_R
        
        c_L = jnp.sqrt(g * jnp.maximum(h_L, 1e-10))
        c_R = jnp.sqrt(g * jnp.maximum(h_R, 1e-10))
        alpha = jnp.maximum(jnp.abs(v_L) + c_L, jnp.abs(v_R) + c_R)
        
        flux_h  = 0.5 * (g1_L + g1_R - alpha * (q1_R - q1_L))
        flux_hu = 0.5 * (g2_L + g2_R - alpha * (q2_R - q2_L))
        flux_hv = 0.5 * (g3_L + g3_R - alpha * (q3_R - q3_L))
        
        return flux_h, flux_hu, flux_hv
    
    def _compute_rhs(self, state: FVPeriodicState) -> Tuple:
        """Compute spatial RHS"""
        h, u, v = state.h, state.u, state.v
        
        # Periodic BCs
        h_ext = jnp.pad(h, ((1, 1), (1, 1)), mode='wrap')
        u_ext = jnp.pad(u, ((1, 1), (1, 1)), mode='wrap')
        v_ext = jnp.pad(v, ((1, 1), (1, 1)), mode='wrap')
        
        # X-fluxes
        h_L = h_ext[:-1, 1:-1]
        h_R = h_ext[1:,  1:-1]
        u_L = u_ext[:-1, 1:-1]
        u_R = u_ext[1:,  1:-1]
        v_L = v_ext[:-1, 1:-1]
        v_R = v_ext[1:,  1:-1]
        
        flux_h_x, flux_hu_x, flux_hv_x = self._rusanov_flux_x(
            h_L, h_R, u_L, u_R, v_L, v_R, self.g
        )
        
        # Y-fluxes
        h_L = h_ext[1:-1, :-1]
        h_R = h_ext[1:-1, 1:]
        u_L = u_ext[1:-1, :-1]
        u_R = u_ext[1:-1, 1:]
        v_L = v_ext[1:-1, :-1]
        v_R = v_ext[1:-1, 1:]
        
        flux_h_y, flux_hu_y, flux_hv_y = self._rusanov_flux_y(
            h_L, h_R, u_L, u_R, v_L, v_R, self.g
        )
        
        # Finite volume update
        dh_dt = -(flux_h_x[1:, :] - flux_h_x[:-1, :]) / self.dx \
                -(flux_h_y[:, 1:] - flux_h_y[:, :-1]) / self.dy
        
        dhu_dt = -(flux_hu_x[1:, :] - flux_hu_x[:-1, :]) / self.dx \
                 -(flux_hu_y[:, 1:] - flux_hu_y[:, :-1]) / self.dy
        
        dhv_dt = -(flux_hv_x[1:, :] - flux_hv_x[:-1, :]) / self.dx \
                 -(flux_hv_y[:, 1:] - flux_hv_y[:, :-1]) / self.dy
        
        # Convert to velocity
        h_safe = jnp.maximum(h, 1e-10)
        du_dt = (dhu_dt - u * dh_dt) / h_safe + self.f * v
        dv_dt = (dhv_dt - v * dh_dt) / h_safe - self.f * u
        
        return dh_dt, du_dt, dv_dt
    
    @jax.jit
    def step(self, state: FVPeriodicState, dt: float) -> FVPeriodicState:
        """RK2 time step"""
        # Stage 1
        dh1, du1, dv1 = self._compute_rhs(state)
        h1 = state.h + dt * dh1
        u1 = state.u + dt * du1
        v1 = state.v + dt * dv1
        
        state1 = FVPeriodicState(h=h1, u=u1, v=v1, 
                                 time=state.time + dt, step=state.step)
        
        # Stage 2
        dh2, du2, dv2 = self._compute_rhs(state1)
        h_new = 0.5 * (state.h + h1 + dt * dh2)
        u_new = 0.5 * (state.u + u1 + dt * du2)
        v_new = 0.5 * (state.v + v1 + dt * dv2)
        
        return FVPeriodicState(
            h=h_new, u=u_new, v=v_new,
            time=state.time + dt,
            step=state.step + 1
        )
    
    def get_diagnostics(self, state: FVPeriodicState) -> Dict[str, float]:
        """Compute conservation diagnostics"""
        h, u, v = state.h, state.u, state.v
        
        mass = float(jnp.sum(h) * self.dx * self.dy)
        ke = float(0.5 * jnp.sum(h * (u**2 + v**2)) * self.dx * self.dy)
        pe = float(0.5 * self.g * jnp.sum(h**2) * self.dx * self.dy)
        energy = ke + pe
        
        return {
            'mass': mass,
            'energy': energy,
            'kinetic_energy': ke,
            'potential_energy': pe,
            'h_min': float(jnp.min(h)),
            'h_max': float(jnp.max(h)),
            'u_max': float(jnp.max(jnp.abs(u))),
            'v_max': float(jnp.max(jnp.abs(v))),
        }
    
    # ========================================================================
    # FRAMEWORK INTERFACE METHODS (new for Framework integration)
    # ========================================================================
    
    def get_available_outputs(self) -> Dict[str, List[str]]:
        """Define available output groups and their variables."""
        return {
            'state': ['h', 'u', 'v'],                           # Full state (for viz/restart)
            'diagnostics': ['mass', 'energy', 'kinetic_energy', 
                          'potential_energy', 'h_min', 'h_max',
                          'u_max', 'v_max']                     # Scalar diagnostics
        }
    
    def get_output_spec(self, config: Dict[str, Any]) -> Dict[str, OutputSpec]:
        """Create output specifications from config."""
        io_config = config['io']['output']
        available = self.get_available_outputs()
        
        return {
            'state': OutputSpec(
                variables=available['state'],
                frequency=io_config['state']['frequency'],
                enabled=io_config['state']['enabled'],
                description="Full state fields (h, u, v)"
            ),
            'diagnostics': OutputSpec(
                variables=available['diagnostics'],
                frequency=io_config['diagnostics']['frequency'],
                enabled=io_config['diagnostics']['enabled'],
                description="Conservation diagnostics"
            )
        }
    
    def state_to_output(self, state: FVPeriodicState, 
                       output_group: str) -> Dict[str, np.ndarray]:
        """Convert state to output format for a specific group."""
        if output_group == 'state':
            return {
                'h': np.array(state.h),
                'u': np.array(state.u),
                'v': np.array(state.v),
                'time': state.time,
                'step': state.step
            }
        elif output_group == 'diagnostics':
            return self.get_diagnostics(state)
        else:
            raise ValueError(f"Unknown output group: {output_group}")
    
    def state_from_checkpoint(self, checkpoint_data: Dict[str, np.ndarray]) -> FVPeriodicState:
        """Restore state from checkpoint data."""
        return FVPeriodicState(
            h=jnp.array(checkpoint_data['h']),
            u=jnp.array(checkpoint_data['u']),
            v=jnp.array(checkpoint_data['v']),
            time=float(checkpoint_data['time']),
            step=int(checkpoint_data['step'])
        )


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("This is an example solver.")
    print("Run via Framework:")
    print("  python -m Framework.runner Examples/fv_periodic_2d/config.yaml")

