"""
Flux-form cubed-sphere FV-PLR shallow water solver (prototype).

Uses the framework interface; reuses halo_exchange helpers for future expansion.
"""

from dataclasses import dataclass
from typing import Dict, List

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

    def step(self, state: SWEState, dt: float) -> SWEState:
        """
        Forward Euler step for flux-form SWE.
        
        Implements:
            ∂h/∂t = -(1/(R√G)) [∂(hu¹√G)/∂ξ¹ + ∂(hu²√G)/∂ξ²]
            ∂(hu^i)/∂t = -(1/(R√G)) ∂(Flux^i √G)/∂ξ^i
        
        where:
            - R = sphere radius [m]
            - dx = grid spacing in computational coordinates [radians]
            - √G = metric Jacobian (unitless)
            - Scale factor: 1/(R*dx) converts ∂/∂ξ to physical derivative [1/m]
        """
        g = self.g
        R = self.planet.R_sphere
        dx = self.geometry.dx  # radians
        sqrtG = self.geometry.sqrtG  # (6, N, N) interior only
        N = self.N
        
        # Extract state (6, N, N)
        h = state.h
        hu1 = state.hu1
        hu2 = state.hu2
        
        # Halo exchange to get ghosted fields (6, N+2, N+2)
        h_ghosts = exchange_scalar_halos_v2(h, N, self.halo_exchange)
        hu1_ghosts = exchange_scalar_halos_v2(hu1, N, self.halo_exchange)
        hu2_ghosts = exchange_scalar_halos_v2(hu2, N, self.halo_exchange)
        
        # Scale factor for converting computational to physical derivatives
        scale = 1.0 / (R * dx)  # [1/m]
        
        # Compute tendencies face by face using 3-point stencils
        dh_dt_all = jnp.zeros((6, N, N))
        dhu1_dt_all = jnp.zeros((6, N, N))
        dhu2_dt_all = jnp.zeros((6, N, N))
        
        for face in range(6):
            # Extract ghosted fields for this face
            h_f = h_ghosts[face]      # (N+2, N+2)
            hu1_f = hu1_ghosts[face]  # (N+2, N+2)
            hu2_f = hu2_ghosts[face]  # (N+2, N+2)
            sqrtG_f = sqrtG[face]     # (N, N) - interior only
            
            # Mass conservation: ∂h/∂t = -(1/(R√G)) [∂(hu¹√G)/∂ξ¹ + ∂(hu²√G)/∂ξ²]
            # 
            # For now, assume √G ≈ constant and ignore metric terms (will fix later)
            # This gives: ∂h/∂t ≈ -(1/R) [∂(hu¹)/∂ξ¹ + ∂(hu²)/∂ξ²]
            # 
            # Using centered differences on ghosted data:
            dhu1_dxi1 = (hu1_f[2:N+2, 1:N+1] - hu1_f[0:N, 1:N+1]) / 2.0  # (N, N)
            dhu2_dxi2 = (hu2_f[1:N+1, 2:N+2] - hu2_f[1:N+1, 0:N]) / 2.0  # (N, N)
            
            dh_dt_all = dh_dt_all.at[face].set(-scale * (dhu1_dxi1 + dhu2_dxi2))
            
            # Momentum (ξ¹): ∂(hu¹)/∂t = -(1/(R√G)) ∂[(hu¹·u¹ + 0.5gh²)√G]/∂ξ¹
            # 
            # Again ignoring metric variations: ∂(hu¹)/∂t ≈ -(1/R) ∂[(hu¹·u¹ + 0.5gh²)]/∂ξ¹
            # 
            # Compute momentum flux at all ghost points (need for centered diff)
            u1_f = hu1_f / h_f  # (N+2, N+2) - velocity at all points including ghosts
            momentum_flux1_f = hu1_f * u1_f + 0.5 * g * h_f * h_f  # (N+2, N+2)
            
            # Centered difference
            dflux1_dxi1 = (momentum_flux1_f[2:N+2, 1:N+1] - momentum_flux1_f[0:N, 1:N+1]) / 2.0  # (N, N)
            dhu1_dt_all = dhu1_dt_all.at[face].set(-scale * dflux1_dxi1)
            
            # Momentum (ξ²): ∂(hu²)/∂t = -(1/(R√G)) ∂[(hu²·u² + 0.5gh²)√G]/∂ξ²
            u2_f = hu2_f / h_f  # (N+2, N+2)
            momentum_flux2_f = hu2_f * u2_f + 0.5 * g * h_f * h_f  # (N+2, N+2)
            
            # Centered difference
            dflux2_dxi2 = (momentum_flux2_f[1:N+1, 2:N+2] - momentum_flux2_f[1:N+1, 0:N]) / 2.0  # (N, N)
            dhu2_dt_all = dhu2_dt_all.at[face].set(-scale * dflux2_dxi2)
        
        # Forward Euler update
        h_new = h + dt * dh_dt_all
        hu1_new = hu1 + dt * dhu1_dt_all
        hu2_new = hu2 + dt * dhu2_dt_all

        if self.sharding is not None:
            h_new = jax.device_put(h_new, self.sharding)
            hu1_new = jax.device_put(hu1_new, self.sharding)
            hu2_new = jax.device_put(hu2_new, self.sharding)

        return SWEState(h=h_new, hu1=hu1_new, hu2=hu2_new,
                        time=state.time + dt, step=state.step + 1)

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

