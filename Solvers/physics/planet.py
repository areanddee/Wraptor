"""
Planet Physical Parameters

This module defines physical constants for planetary simulations.
All values defined here - no magic numbers scattered in solver code!

Usage:
    # From config file
    planet = PlanetParams.from_config(config)
    
    # Use preset
    planet = EARTH
    planet = MARS
    
    # Custom planet
    planet = PlanetParams(R_sphere=3.39e6, gravity=3.71, omega=7.088e-5)
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class PlanetParams:
    """
    Physical parameters for a planet.
    
    Immutable after creation (frozen dataclass).
    
    Attributes:
        R_sphere: Mean radius [m]
        gravity: Surface gravity [m/s²]
        omega: Rotation rate [rad/s]
        name: Planet name (for logging)
    
    Derived quantities (as properties):
        day_length: Sidereal day length [s]
        circumference: Equatorial circumference [m]
    """
    R_sphere: float = 6.371e6      # Earth default
    gravity: float = 9.81          # Earth default
    omega: float = 7.292e-5        # Earth default
    name: str = "Custom"
    
    @property
    def day_length(self) -> float:
        """Sidereal day length in seconds."""
        if self.omega == 0:
            return float('inf')
        return 2 * 3.14159265359 / self.omega
    
    @property
    def circumference(self) -> float:
        """Equatorial circumference in meters."""
        return 2 * 3.14159265359 * self.R_sphere
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PlanetParams':
        """
        Create PlanetParams from configuration dictionary.
        
        Looks for 'physics' section in config. Falls back to Earth defaults.
        
        Args:
            config: Configuration dictionary (typically from YAML)
            
        Returns:
            PlanetParams instance
            
        Example config:
            physics:
              R_sphere: 6.371e6
              gravity: 9.81
              omega: 7.292e-5
              name: Earth
        """
        physics = config.get('physics', {})
        
        return cls(
            R_sphere=float(physics.get('R_sphere', 6.371e6)),
            gravity=float(physics.get('gravity', 9.81)),
            omega=float(physics.get('omega', 7.292e-5)),
            name=str(physics.get('name', 'Earth'))
        )
    
    def __str__(self) -> str:
        return (f"PlanetParams({self.name}: R={self.R_sphere/1e6:.3f}×10⁶ m, "
                f"g={self.gravity:.2f} m/s², ω={self.omega:.3e} rad/s)")


# ============================================================================
# PRESETS
# ============================================================================

EARTH = PlanetParams(
    R_sphere=6.371e6,
    gravity=9.81,
    omega=7.292e-5,
    name="Earth"
)

MARS = PlanetParams(
    R_sphere=3.3895e6,
    gravity=3.71,
    omega=7.088e-5,  # 24h 37m rotation
    name="Mars"
)

# Non-rotating planet (for testing)
STATIC_EARTH = PlanetParams(
    R_sphere=6.371e6,
    gravity=9.81,
    omega=0.0,
    name="Static Earth"
)

# Small test planet (for debugging)
SMALL_PLANET = PlanetParams(
    R_sphere=1.0e6,
    gravity=1.0,
    omega=1.0e-4,
    name="Small Test Planet"
)

