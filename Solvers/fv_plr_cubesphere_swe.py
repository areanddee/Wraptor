"""
Flux-form cubed-sphere FV-PLR shallow water solver.

Implements:
- Piecewise Linear Reconstruction (PLR) with MC limiter
- RK3 time integration
- Conservative flux-form discretization
- Full metric terms (√G)
- Proper halo exchange at each RK stage

Adapted from fv_plr_cubesphere_adv.py template.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from Framework.solver_interface import NumericalSolver, OutputSpec, SolverState
from Solvers.geometry import CubedSphereGeometry
from Solvers.initial_conditions.swe_testcase2 import steady_geostrophic_flow
from Solvers.physics import PlanetParams
from Solvers.halo_exchange import (
    create_communication_schedule,
    make_halo_exchange,
    exchange_scalar_halos_v2
)


@struct.dataclass
class SWEState(SolverState):
    h: jnp.ndarray
    hu1: jnp.ndarray
    hu2: jnp.ndarray
    time: float
    step: int


class CubedSphereSWE(NumericalSolver):
    def __init__(self, N: int, config: Dict):
        super().__init__()  # NumericalSolver (ABC) has no __init__, just call object.__init__()
        self.N = N
        self.config = config  # Store config for later use
        self.geometry = CubedSphereGeometry.create(N)
        self.planet = PlanetParams.from_config(config)

        self.mu = self.planet.omega
        self.g = self.planet.gravity

        schedule = create_communication_schedule()
        self.halo_exchange = make_halo_exchange(schedule, N)

        para_config = config.get("parallelization", {})
        enable_sharding = para_config.get("enable_sharding", False)
        if enable_sharding:
            self.setup_sharding(para_config)
        else:
            self.sharding = None

    def setup_sharding(self, para_config: Dict):
        num_devices = para_config.get("num_devices", 6)
        tiles_per_edge = para_config.get("tiles_per_edge", 1)
        num_tiles = 6 * tiles_per_edge * tiles_per_edge
        if num_tiles % num_devices != 0:
            raise ValueError("num_tiles must be divisible by num_devices")

        devices = jax.devices()[:num_devices]
        mesh = jax.sharding.Mesh(devices, ('tiles',))
        self.sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('tiles'))

    def initialize(self, config: Dict = None) -> SWEState:
        h, u_lon, u_lat = steady_geostrophic_flow(
            self.geometry, self.planet,
            h0=config.get("physics", {}).get("h0", 8000.0),
            u0=config.get("physics", {}).get("u0", 40.0)
        )

        hu1 = jnp.array(h * u_lon)
        hu2 = jnp.array(h * u_lat)

        if self.sharding is not None:
            h = jax.device_put(h, self.sharding)
            hu1 = jax.device_put(hu1, self.sharding)
            hu2 = jax.device_put(hu2, self.sharding)

        return SWEState(h=h, hu1=hu1, hu2=hu2, time=0.0, step=0)

    # ========================================================================
    # PLR HELPER FUNCTIONS
    # ========================================================================

    @staticmethod
    def minmod_limiter(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        """
        Monotonized Central (MC) limiter for PLR reconstruction.
        
        Returns minmod(2a, (a+b)/2, 2b) as in van Leer (1977).
        """
        return jnp.where(
            a * b <= 0, 0.0,
            jnp.where(
                jnp.abs(a) < jnp.abs(b),
                jnp.where(jnp.abs(a) < jnp.abs(c), a, c),
                jnp.where(jnp.abs(b) < jnp.abs(c), b, c)
            )
        )

    @staticmethod
    def compute_limited_slopes(field_ghosts: jnp.ndarray, dx: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute MC-limited slopes for PLR reconstruction.
        
        Args:
            field_ghosts: (N+2, N+2) with ghost cells
            dx: grid spacing [radians]
        
        Returns:
            slope1: (N, N) limited slope in ξ¹ direction
            slope2: (N, N) limited slope in ξ² direction
        """
        # Extract interior for slope computation
        f = field_ghosts[1:-1, 1:-1]  # (N, N)
        
        # ξ¹ direction slopes
        f_im1 = field_ghosts[0:-2, 1:-1]
        f_ip1 = field_ghosts[2:, 1:-1]
        
        grad_central_1 = (f_ip1 - f_im1) / (2.0 * dx)
        grad_forward_1 = (f_ip1 - f) / dx
        grad_backward_1 = (f - f_im1) / dx
        
        slope1 = CubedSphereSWE.minmod_limiter(
            2.0 * grad_backward_1,
            grad_central_1,
            2.0 * grad_forward_1
        )
        
        # ξ² direction slopes
        f_jm1 = field_ghosts[1:-1, 0:-2]
        f_jp1 = field_ghosts[1:-1, 2:]
        
        grad_central_2 = (f_jp1 - f_jm1) / (2.0 * dx)
        grad_forward_2 = (f_jp1 - f) / dx
        grad_backward_2 = (f - f_jm1) / dx
        
        slope2 = CubedSphereSWE.minmod_limiter(
            2.0 * grad_backward_2,
            grad_central_2,
            2.0 * grad_forward_2
        )
        
        return slope1, slope2

    def compute_swe_flux_plr_face(self, h_ghosts: jnp.ndarray, hu1_ghosts: jnp.ndarray, 
                                  hu2_ghosts: jnp.ndarray, sqrtG_ghosts: jnp.ndarray,
                                  dx: float, N: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, 
                                                                jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute SWE fluxes using PLR reconstruction for one face.
        
        Returns mass flux and momentum fluxes in both directions.
        
        Args:
            h_ghosts: (N+2, N+2) depth with ghosts
            hu1_ghosts: (N+2, N+2) momentum hu¹ with ghosts
            hu2_ghosts: (N+2, N+2) momentum hu² with ghosts
            sqrtG_ghosts: (N+2, N+2) metric with ghosts
            dx: grid spacing [radians]
            N: interior resolution
        
        Returns:
            Fmass1: (N+1, N) mass flux in ξ¹ direction
            Fmass2: (N, N+1) mass flux in ξ² direction
            Fmom1_1: (N+1, N) momentum1 flux in ξ¹ direction
            Fmom1_2: (N, N+1) momentum1 flux in ξ² direction
            Fmom2_1: (N+1, N) momentum2 flux in ξ¹ direction
            Fmom2_2: (N, N+1) momentum2 flux in ξ² direction
        """
        g = self.g
        
        # Compute limited slopes for h, hu1, hu2
        h_slope1, h_slope2 = self.compute_limited_slopes(h_ghosts, dx)
        hu1_slope1, hu1_slope2 = self.compute_limited_slopes(hu1_ghosts, dx)
        hu2_slope1, hu2_slope2 = self.compute_limited_slopes(hu2_ghosts, dx)
        
        # Interior values
        h_int = h_ghosts[1:-1, 1:-1]  # (N, N)
        hu1_int = hu1_ghosts[1:-1, 1:-1]
        hu2_int = hu2_ghosts[1:-1, 1:-1]
        
        # ξ¹ direction fluxes (between cells i and i+1)
        # Left edge of cell (i+1): reconstruct from cell i
        h_right = h_int + 0.5 * dx * h_slope1
        hu1_right = hu1_int + 0.5 * dx * hu1_slope1
        hu2_right = hu2_int + 0.5 * dx * hu2_slope1
        
        # Right edge of cell i: reconstruct from cell (i+1)
        h_left = h_int - 0.5 * dx * h_slope1
        hu1_left = hu1_int - 0.5 * dx * hu1_slope1
        hu2_left = hu2_int - 0.5 * dx * hu2_slope1
        
        # Concatenate with ghost values to get (N+1, N) edge arrays
        h_right_edge = jnp.concatenate([h_ghosts[0:1, 1:N+1], h_right, h_ghosts[N+1:N+2, 1:N+1]], axis=0)
        h_left_edge = jnp.concatenate([h_ghosts[0:1, 1:N+1], h_left, h_ghosts[N+1:N+2, 1:N+1]], axis=0)
        
        hu1_right_edge = jnp.concatenate([hu1_ghosts[0:1, 1:N+1], hu1_right, hu1_ghosts[N+1:N+2, 1:N+1]], axis=0)
        hu1_left_edge = jnp.concatenate([hu1_ghosts[0:1, 1:N+1], hu1_left, hu1_ghosts[N+1:N+2, 1:N+1]], axis=0)
        
        hu2_right_edge = jnp.concatenate([hu2_ghosts[0:1, 1:N+1], hu2_right, hu2_ghosts[N+1:N+2, 1:N+1]], axis=0)
        hu2_left_edge = jnp.concatenate([hu2_ghosts[0:1, 1:N+1], hu2_left, hu2_ghosts[N+1:N+2, 1:N+1]], axis=0)
        
        # Compute velocities at edges
        u1_right_edge = hu1_right_edge / h_right_edge
        u1_left_edge = hu1_left_edge / h_left_edge
        
        # Upwind selection for mass flux
        u1_face = 0.5 * (u1_right_edge[1:N+2, :] + u1_left_edge[0:N+1, :])  # Average velocity at face
        h_face = jnp.where(u1_face > 0, h_right_edge[0:N+1, :], h_left_edge[1:N+2, :])
        
        # Average √G at faces
        sqrtG_face1 = 0.5 * (sqrtG_ghosts[1:N+2, 1:N+1] + sqrtG_ghosts[0:N+1, 1:N+1])
        
        # Mass flux
        Fmass1 = sqrtG_face1 * u1_face * h_face  # (N+1, N)
        
        # Momentum fluxes (ξ¹ direction)
        hu1_face = jnp.where(u1_face > 0, hu1_right_edge[0:N+1, :], hu1_left_edge[1:N+2, :])
        pressure_face = 0.5 * g * h_face * h_face
        Fmom1_1 = sqrtG_face1 * (u1_face * hu1_face + pressure_face)  # (N+1, N)
        
        hu2_face = jnp.where(u1_face > 0, hu2_right_edge[0:N+1, :], hu2_left_edge[1:N+2, :])
        Fmom2_1 = sqrtG_face1 * u1_face * hu2_face  # (N+1, N)
        
        # ξ² direction fluxes (similar logic)
        h_top = h_int + 0.5 * dx * h_slope2
        hu1_top = hu1_int + 0.5 * dx * hu1_slope2
        hu2_top = hu2_int + 0.5 * dx * hu2_slope2
        
        h_bottom = h_int - 0.5 * dx * h_slope2
        hu1_bottom = hu1_int - 0.5 * dx * hu1_slope2
        hu2_bottom = hu2_int - 0.5 * dx * hu2_slope2
        
        h_top_edge = jnp.concatenate([h_ghosts[1:N+1, 0:1], h_top, h_ghosts[1:N+1, N+1:N+2]], axis=1)
        h_bottom_edge = jnp.concatenate([h_ghosts[1:N+1, 0:1], h_bottom, h_ghosts[1:N+1, N+1:N+2]], axis=1)
        
        hu1_top_edge = jnp.concatenate([hu1_ghosts[1:N+1, 0:1], hu1_top, hu1_ghosts[1:N+1, N+1:N+2]], axis=1)
        hu1_bottom_edge = jnp.concatenate([hu1_ghosts[1:N+1, 0:1], hu1_bottom, hu1_ghosts[1:N+1, N+1:N+2]], axis=1)
        
        hu2_top_edge = jnp.concatenate([hu2_ghosts[1:N+1, 0:1], hu2_top, hu2_ghosts[1:N+1, N+1:N+2]], axis=1)
        hu2_bottom_edge = jnp.concatenate([hu2_ghosts[1:N+1, 0:1], hu2_bottom, hu2_ghosts[1:N+1, N+1:N+2]], axis=1)
        
        u2_top_edge = hu2_top_edge / h_top_edge
        u2_bottom_edge = hu2_bottom_edge / h_bottom_edge
        
        u2_face = 0.5 * (u2_top_edge[:, 1:N+2] + u2_bottom_edge[:, 0:N+1])
        h_face2 = jnp.where(u2_face > 0, h_top_edge[:, 0:N+1], h_bottom_edge[:, 1:N+2])
        
        sqrtG_face2 = 0.5 * (sqrtG_ghosts[1:N+1, 1:N+2] + sqrtG_ghosts[1:N+1, 0:N+1])
        
        Fmass2 = sqrtG_face2 * u2_face * h_face2  # (N, N+1)
        
        hu1_face2 = jnp.where(u2_face > 0, hu1_top_edge[:, 0:N+1], hu1_bottom_edge[:, 1:N+2])
        Fmom1_2 = sqrtG_face2 * u2_face * hu1_face2  # (N, N+1)
        
        hu2_face2 = jnp.where(u2_face > 0, hu2_top_edge[:, 0:N+1], hu2_bottom_edge[:, 1:N+2])
        pressure_face2 = 0.5 * g * h_face2 * h_face2
        Fmom2_2 = sqrtG_face2 * (u2_face * hu2_face2 + pressure_face2)  # (N, N+1)
        
        return Fmass1, Fmass2, Fmom1_1, Fmom1_2, Fmom2_1, Fmom2_2

    def step(self, state: SWEState, dt: float) -> SWEState:
        """
        RK3 (TVD) step for flux-form SWE with PLR reconstruction and Coriolis.
        
        Implements third-order Runge-Kutta timestepping:
            k1 = RHS(state0)
            k2 = RHS(state0 + dt*k1)
            k3 = RHS(state0 + dt/4*(k1 + k2))
            state_new = state0 + dt/6*(k1 + k2 + 4*k3)
        """
        N = self.N
        dx = self.geometry.dx
        R = self.planet.R_sphere
        sqrtG = self.geometry.sqrtG  # (6, N, N)
        
        # Extend sqrt(G) to include ghosts
        sqrtG_ghosts = jnp.pad(sqrtG, ((0, 0), (1, 1), (1, 1)), mode='edge')  # (6, N+2, N+2)
        
        # Coriolis parameter f = 2Ω sin(φ) at each cell center
        lat_lon = self.geometry.get_lat_lon_all_faces()  # Returns (lat, lon) both (6, N, N)
        lat = lat_lon[0]  # (6, N, N)
        f = 2.0 * self.planet.omega * jnp.sin(lat)  # (6, N, N)
        
        # Helper: compute RHS for all faces
        def compute_rhs(h, hu1, hu2):
            # Exchange halos
            h_ghosts = exchange_scalar_halos_v2(h, N, self.halo_exchange)
            hu1_ghosts = exchange_scalar_halos_v2(hu1, N, self.halo_exchange)
            hu2_ghosts = exchange_scalar_halos_v2(hu2, N, self.halo_exchange)
            
            dh_dt = jnp.zeros((6, N, N))
            dhu1_dt = jnp.zeros((6, N, N))
            dhu2_dt = jnp.zeros((6, N, N))
            
            for face in range(6):
                # Compute fluxes
                Fm1, Fm2, Fmu1_1, Fmu1_2, Fmu2_1, Fmu2_2 = self.compute_swe_flux_plr_face(
                    h_ghosts[face], hu1_ghosts[face], hu2_ghosts[face],
                    sqrtG_ghosts[face], dx, N
                )
                
                # Flux divergence
                div_Fmass = (Fm1[1:, :] - Fm1[:-1, :]) + (Fm2[:, 1:] - Fm2[:, :-1])
                div_Fmu1 = (Fmu1_1[1:, :] - Fmu1_1[:-1, :]) + (Fmu1_2[:, 1:] - Fmu1_2[:, :-1])
                div_Fmu2 = (Fmu2_1[1:, :] - Fmu2_1[:-1, :]) + (Fmu2_2[:, 1:] - Fmu2_2[:, :-1])
                
                # RHS = -(1/(R·dx·√G)) · div(F) + Coriolis
                scale = 1.0 / (R * dx)
                dh_dt = dh_dt.at[face].set(-scale * div_Fmass / sqrtG[face])
                
                # Coriolis force: f·hu² for ξ¹ momentum, -f·hu¹ for ξ² momentum
                dhu1_dt = dhu1_dt.at[face].set(-scale * div_Fmu1 / sqrtG[face] + f[face] * hu2[face])
                dhu2_dt = dhu2_dt.at[face].set(-scale * div_Fmu2 / sqrtG[face] - f[face] * hu1[face])
            
            return dh_dt, dhu1_dt, dhu2_dt
        
        # RK3 stages
        h0, hu1_0, hu2_0 = state.h, state.hu1, state.hu2
        
        # Stage 1
        k1_h, k1_hu1, k1_hu2 = compute_rhs(h0, hu1_0, hu2_0)
        h1 = h0 + dt * k1_h
        hu1_1 = hu1_0 + dt * k1_hu1
        hu2_1 = hu2_0 + dt * k1_hu2
        
        # Stage 2
        k2_h, k2_hu1, k2_hu2 = compute_rhs(h1, hu1_1, hu2_1)
        h2 = h0 + 0.25 * dt * (k1_h + k2_h)
        hu1_2 = hu1_0 + 0.25 * dt * (k1_hu1 + k2_hu1)
        hu2_2 = hu2_0 + 0.25 * dt * (k1_hu2 + k2_hu2)
        
        # Stage 3
        k3_h, k3_hu1, k3_hu2 = compute_rhs(h2, hu1_2, hu2_2)
        h_new = h0 + (dt / 6.0) * (k1_h + k2_h + 4.0 * k3_h)
        hu1_new = hu1_0 + (dt / 6.0) * (k1_hu1 + k2_hu1 + 4.0 * k3_hu1)
        hu2_new = hu2_0 + (dt / 6.0) * (k1_hu2 + k2_hu2 + 4.0 * k3_hu2)
        
        if self.sharding is not None:
            h_new = jax.device_put(h_new, self.sharding)
            hu1_new = jax.device_put(hu1_new, self.sharding)
            hu2_new = jax.device_put(hu2_new, self.sharding)
        
        return SWEState(h=h_new, hu1=hu1_new, hu2=hu2_new, time=state.time + dt, step=state.step + 1)

    def get_diagnostics(self, state: SWEState) -> Dict[str, float]:
        h = state.h
        hu1 = state.hu1
        hu2 = state.hu2
        u1 = hu1 / h
        u2 = hu2 / h
        mass = float(jnp.sum(h * self.geometry.sqrtG))
        ke = 0.5 * jnp.sum(h * (u1 ** 2 + u2 ** 2) * self.geometry.sqrtG)
        geop = 0.5 * self.g * jnp.sum(h ** 2 * self.geometry.sqrtG)
        return {"mass": mass, "kinetic": float(ke), "geopotential": float(geop)}

    def get_available_outputs(self) -> Dict[str, List[str]]:
        return {
            "state": ["h", "hu1", "hu2"],
            "diagnostics": ["mass", "kinetic", "geopotential"],
        }

    def get_output_spec(self, config: Dict) -> Dict[str, OutputSpec]:
        output_cfg = config.get("io", {}).get("output", {})
        specs: Dict[str, OutputSpec] = {}
        for group, vars in self.get_available_outputs().items():
            grp_cfg = output_cfg.get(group, {})
            if grp_cfg.get("enabled", False):
                specs[group] = OutputSpec(frequency=grp_cfg.get("frequency", 10), variables=vars)
        return specs

    def state_to_output(self, state: SWEState, output_group: str) -> Dict[str, jnp.ndarray]:
        if output_group == "state":
            return {"h": state.h, "hu1": state.hu1, "hu2": state.hu2, "time": state.time, "step": state.step}
        elif output_group == "diagnostics":
            return self.get_diagnostics(state)
        else:
            raise ValueError(f"Unknown output group {output_group}")

    def state_from_checkpoint(self, checkpoint_data: Dict) -> SWEState:
        h = checkpoint_data["h"]
        hu1 = checkpoint_data["hu1"]
        hu2 = checkpoint_data["hu2"]
        time = float(checkpoint_data["time"])
        step = int(checkpoint_data["step"])
        if self.sharding is not None:
            h = jax.device_put(h, self.sharding)
            hu1 = jax.device_put(hu1, self.sharding)
            hu2 = jax.device_put(hu2, self.sharding)
        return SWEState(h=h, hu1=hu1, hu2=hu2, time=time, step=step)

