"""
Solver Interface for JaxStream2 Framework

This module defines the abstract base class that all solvers must inherit from.
It enforces a consistent contract for initialization, time-stepping, I/O, and restart.

Design Principles:
1. Solvers control WHAT and WHEN to output (physics-driven)
2. Framework controls HOW to save/load (infrastructure)
3. Clear separation: physics (solver) vs. mechanics (framework)

Author: JaxStream2 Framework
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Protocol
from dataclasses import dataclass
import numpy as np


# ============================================================================
# PROTOCOL: Solver State
# ============================================================================

class SolverState(Protocol):
    """
    Protocol for solver state objects.
    
    All solver states must have time and step tracking.
    Use @struct.dataclass from flax for JAX compatibility.
    """
    time: float
    step: int


# ============================================================================
# OUTPUT SPECIFICATION
# ============================================================================

@dataclass
class OutputSpec:
    """
    Specification for a single output stream.
    
    Defines what variables to save and how often.
    Solver creates these, IOManager uses them.
    
    Attributes:
        variables: List of variable names to save (e.g., ['T', 'u', 'v'])
        frequency: Save every N timesteps
        enabled: Whether this output stream is active
        description: Human-readable description for metadata
    """
    variables: List[str]
    frequency: int
    enabled: bool = True
    description: str = ""
    
    def should_output(self, step: int) -> bool:
        """Check if this stream should output at given step."""
        return self.enabled and (step % self.frequency == 0)


# ============================================================================
# ABSTRACT SOLVER BASE CLASS
# ============================================================================

class NumericalSolver(ABC):
    """
    Abstract base class for all numerical solvers.
    
    All production solvers MUST inherit from this class and implement
    all abstract methods. This ensures consistent behavior and enables
    generic framework operations (I/O, restart, orchestration).
    
    Workflow:
        1. Solver is instantiated with config
        2. initialize() creates initial state
        3. step() advances state forward in time (called repeatedly)
        4. get_diagnostics() computes monitoring quantities
        5. Framework uses get_output_spec() to determine what/when to save
        6. Framework calls state_to_output() to get saveable data
        7. For restart: Framework loads data, calls state_from_checkpoint()
    
    Example:
        class MyAwesomeSolver(NumericalSolver):
            def __init__(self, config):
                self.N = config['grid']['N']
                # ... setup
            
            def initialize(self, config):
                return MyState(T=initial_field, time=0.0, step=0)
            
            def step(self, state, dt):
                # ... physics ...
                return updated_state
            
            # ... implement other abstract methods ...
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> SolverState:
        """
        Create initial state from configuration.
        
        This method sets up the initial conditions based on the config file.
        It may include analytical solutions, perturbations, or reading from
        external data sources.
        
        Args:
            config: Full configuration dictionary (from YAML)
        
        Returns:
            Initial solver state (must have time=0.0, step=0)
        
        Example:
            def initialize(self, config):
                N = self.N
                ic_type = config.get('initial_condition', 'cosine_bell')
                
                if ic_type == 'cosine_bell':
                    q = create_cosine_bell(N)
                elif ic_type == 'gaussian':
                    q = create_gaussian(N)
                
                return MyState(q=q, time=0.0, step=0)
        """
        pass
    
    @abstractmethod
    def step(self, state: SolverState, dt: float) -> SolverState:
        """
        Advance state forward by one timestep.
        
        This is the core physics routine. It should:
        - Take current state and timestep
        - Compute spatial operators (fluxes, diffusion, sources)
        - Advance state forward in time (RK, Euler, etc.)
        - Return new state with updated time and step
        
        Args:
            state: Current solver state
            dt: Timestep size [seconds]
        
        Returns:
            Updated state at time = state.time + dt
        
        Notes:
            - This method is typically JIT-compiled for performance
            - Should be pure (no side effects)
            - Must increment state.time and state.step
        
        Example:
            @jax.jit
            def step(self, state, dt):
                # Compute RHS
                dq_dt = self.compute_rhs(state)
                
                # Update (forward Euler)
                q_new = state.q + dt * dq_dt
                
                return MyState(
                    q=q_new,
                    time=state.time + dt,
                    step=state.step + 1
                )
        """
        pass
    
    @abstractmethod
    def get_diagnostics(self, state: SolverState) -> Dict[str, float]:
        """
        Compute diagnostic quantities for monitoring.
        
        Returns scalar quantities useful for:
        - Monitoring simulation health (mass, energy conservation)
        - Detecting numerical issues (NaN, overflow)
        - Logging progress (max values, norms)
        
        Args:
            state: Current solver state
        
        Returns:
            Dictionary of diagnostic names → values (all scalars)
        
        Notes:
            - All values must be Python scalars (use float(jnp_array))
            - Called frequently (every few steps), keep it fast
            - Used for logging, not saved to output files
        
        Example:
            def get_diagnostics(self, state):
                return {
                    'mass': float(jnp.sum(state.q * self.sqrtG * self.dx**2)),
                    'q_max': float(jnp.max(state.q)),
                    'q_min': float(jnp.min(state.q)),
                    'time': state.time,
                    'step': state.step
                }
        """
        pass
    
    @abstractmethod
    def get_available_outputs(self) -> Dict[str, List[str]]:
        """
        Define available output streams and their variables.
        
        This method declares what output groups exist and what variables
        each group contains. The Framework uses this to validate config
        and set up I/O managers.
        
        Returns:
            Dictionary mapping output_group_name → list of variable names
        
        Example:
            def get_available_outputs(self):
                return {
                    'state': ['T'],                    # Full temperature field
                    'diagnostics': ['mass', 'energy'], # Scalar diagnostics
                    'geometry': ['xi1', 'xi2', 'sqrtG'] # Grid geometry (once)
                }
        
        Notes:
            - Group names should be descriptive (state, diagnostics, debug, etc.)
            - Variables in 'state' group are used for restart
            - Framework will validate config against this list
        """
        pass
    
    @abstractmethod
    def get_output_spec(self, config: Dict[str, Any]) -> Dict[str, OutputSpec]:
        """
        Create output specifications from config.
        
        Reads the config's I/O section and creates OutputSpec objects
        that tell the Framework what/when to save. Solver controls the
        variable lists (from get_available_outputs), user controls frequency.
        
        Args:
            config: Full configuration dictionary
        
        Returns:
            Dictionary mapping output_group_name → OutputSpec
        
        Example:
            def get_output_spec(self, config):
                io_config = config['io']['output']
                available = self.get_available_outputs()
                
                return {
                    'state': OutputSpec(
                        variables=available['state'],
                        frequency=io_config['state']['frequency'],
                        enabled=io_config['state']['enabled'],
                        description="Full temperature field"
                    ),
                    'diagnostics': OutputSpec(
                        variables=available['diagnostics'],
                        frequency=io_config['diagnostics']['frequency'],
                        enabled=True,
                        description="Conservation diagnostics"
                    )
                }
        """
        pass
    
    @abstractmethod
    def state_to_output(self, state: SolverState, 
                       output_group: str) -> Dict[str, np.ndarray]:
        """
        Convert solver state to output format for a specific group.
        
        Extracts the requested variables from the state and converts them
        to NumPy arrays for saving. Different output groups may include
        different variables at different frequencies.
        
        Args:
            state: Current solver state
            output_group: Name of output group (from get_output_spec keys)
        
        Returns:
            Dictionary mapping variable_name → numpy array
        
        Example:
            def state_to_output(self, state, output_group):
                if output_group == 'state':
                    return {
                        'T': np.array(state.T),
                        'time': state.time,
                        'step': state.step
                    }
                elif output_group == 'diagnostics':
                    diag = self.get_diagnostics(state)
                    return diag  # Already dict of scalars
                else:
                    raise ValueError(f"Unknown output group: {output_group}")
        
        Notes:
            - Must return NumPy arrays (use np.array(jax_array))
            - Framework will validate against get_available_outputs()
            - Only called when OutputSpec.should_output(step) is True
        """
        pass
    
    @abstractmethod
    def state_from_checkpoint(self, checkpoint_data: Dict[str, np.ndarray]) -> SolverState:
        """
        Restore solver state from checkpoint data.
        
        Inverse of state_to_output('state'). Takes saved NumPy arrays
        and reconstructs the solver state for restart capability.
        
        Args:
            checkpoint_data: Dictionary of variable_name → numpy array
                            (loaded from Orbax checkpoint)
        
        Returns:
            Restored solver state
        
        Example:
            def state_from_checkpoint(self, checkpoint_data):
                # Convert numpy → JAX arrays
                T = jnp.array(checkpoint_data['T'])
                time = float(checkpoint_data['time'])
                step = int(checkpoint_data['step'])
                
                # Reconstruct state
                return MyState(T=T, time=time, step=step)
        
        Notes:
            - Must handle both NumPy and JAX arrays gracefully
            - Apply device sharding if solver was initialized with it
            - Validate array shapes match solver configuration
        """
        pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_solver_interface(solver: NumericalSolver) -> bool:
    """
    Validate that a solver implements the required interface.
    
    Args:
        solver: Solver instance to validate
    
    Returns:
        True if valid
    
    Raises:
        TypeError: If required methods are missing
    """
    required_methods = [
        'initialize', 'step', 'get_diagnostics',
        'get_available_outputs', 'get_output_spec',
        'state_to_output', 'state_from_checkpoint'
    ]
    
    for method in required_methods:
        if not hasattr(solver, method):
            raise TypeError(
                f"Solver {solver.__class__.__name__} missing required method: {method}"
            )
        if not callable(getattr(solver, method)):
            raise TypeError(
                f"Solver {solver.__class__.__name__}.{method} is not callable"
            )
    
    return True


# ============================================================================
# DOCUMENTATION
# ============================================================================

__doc__ += """

Quick Start Guide for New Solvers:
==================================

1. Create your solver file in Solvers/:
   
   from Framework.solver_interface import NumericalSolver, OutputSpec
   from flax import struct
   
   @struct.dataclass
   class MyState:
       q: jax.Array
       time: float
       step: int
   
   class MySolver(NumericalSolver):
       def __init__(self, config):
           # Setup grid, parameters, etc.
           pass
       
       # Implement all abstract methods
       # (IDE will show which ones are required)

2. Create a config file in Config/:
   
   solver:
     type: my_solver
     N: 60
   
   io:
     output:
       state:
         frequency: 10
         enabled: true

3. Test your solver:
   
   python -m Framework.runner Config/my_solver_config.yaml

That's it! Framework handles I/O, restart, sharding, etc.
"""

