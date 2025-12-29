# Metric Factors in Cubed-Sphere SWE Solver

## Overview

When solving PDEs on curved manifolds (like the sphere), the metric tensor appears in different ways depending on the operation. A common source of confusion is when √G (the metric determinant) is needed versus the inverse metric g^{ij}.

## Key Distinction: Gradient vs Divergence

| Operation | Mathematical Form | Metric Factor | Physical Meaning |
|-----------|-------------------|---------------|------------------|
| **Gradient** of scalar f | ∇f = gⁱʲ (∂f/∂ξʲ) **e**ᵢ | Inverse metric gⁱʲ | Rate of change in space |
| **Divergence** of vector F | ∇·F = (1/√G) ∂(√G Fⁱ)/∂ξⁱ | √G in flux form | Net outflow per unit volume |
| **Area integral** | ∫∫ f dA = ∫∫ f √G dξ¹dξ² | √G as Jacobian | Physical area element |

## Application to SWE Momentum Equation

The momentum equation in vector-invariant form:

```
∂V/∂t = -∇B + (ζ+f)k×V
```

where B = gh + |V|²/2 is the Bernoulli function (specific energy, J/kg).

### Bernoulli Gradient Term

The Bernoulli function B is a **coordinate-invariant scalar** - it has the same physical value regardless of coordinate system. Its gradient transforms as:

```
∇B = J (JᵀJ)⁻¹ [∂B/∂ξ¹, ∂B/∂ξ²]ᵀ
```

where:
- J is the 3×2 Jacobian matrix ∂(X,Y,Z)/∂(ξ¹,ξ²)
- JᵀJ is the 2×2 covariant metric tensor gᵢⱼ
- (JᵀJ)⁻¹ is the contravariant metric tensor gⁱʲ

**No √G appears in gradient computation** - the inverse metric handles the geometry.

### Mass Flux Divergence (where √G IS needed)

The mass equation:

```
∂h/∂t + ∇·(hV) = 0
```

In coordinate form:

```
∂h/∂t = -(1/√G)[∂(√G u¹ h)/∂ξ¹ + ∂(√G u² h)/∂ξ²]
```

Here √G appears explicitly because we're computing the net flux across cell boundaries, which depends on the physical area of each face.

## Computational Workflow for Bernoulli Gradient

### Step 1: Pre-compute Geometry (once at initialization)

Define the gradient operator matrix M = J(JᵀJ)⁻¹, a 3×2 matrix at each grid point:

```
M = [M₁₁  M₁₂]     where  Mₐᵦ = Σₖ Jₐₖ (JᵀJ)⁻¹ₖᵦ
    [M₂₁  M₂₂]
    [M₃₁  M₃₂]
```

This can be stored as 6 arrays of shape (N, N) per face, or equivalently as separate J and (JᵀJ)⁻¹ components.

### Step 2: At Each Timestep

1. **Compute Bernoulli function** from state variables:
   ```
   B = g·h + ½(Vx² + Vy² + Vz²)
   ```

2. **Differentiate in coordinate space** using FV3s stencils:
   ```
   ∂B/∂ξ¹, ∂B/∂ξ² = compute_slopes_fv3(B, dx, N)
   ```
   
   Key: One-sided stencils at boundaries mean **no ghost cells needed**.

3. **Transform to Cartesian gradient**:
   ```
   [dB/dX]       [∂B/∂ξ¹]
   [dB/dY] = M · [∂B/∂ξ²]
   [dB/dZ]
   ```

4. **Add to momentum RHS**:
   ```
   dVx/dt = ... - dB/dX + ...
   dVy/dt = ... - dB/dY + ...
   dVz/dt = ... - dB/dZ + ...
   ```

## Summary

| Term in SWE | Metric Factor | Reason |
|-------------|---------------|--------|
| ∇B (Bernoulli gradient) | (JᵀJ)⁻¹ via M matrix | Gradient uses inverse metric |
| ∇·(hV) (mass flux) | √G | Divergence uses metric determinant |
| ∫∫ h dA (total mass) | √G | Integration uses area element |

The distinction is fundamental: gradients measure **rate of change** (inverse metric), while divergences measure **flux through surfaces** (metric determinant).
