# GPU Profiling Guide for JAX/XLA Applications

This document covers profiling tools for analyzing performance of JAX-based GPU applications on NVIDIA hardware.

## Overview

| Tool | Purpose | Granularity | Output |
|------|---------|-------------|--------|
| NSight Systems (nsys) | Timeline profiling | System-wide | .nsys-rep |
| NSight Compute (ncu) | Kernel analysis | Per-kernel | .ncu-rep |
| JAX Profiler | JAX operations | XLA HLO level | Chrome trace |
| JAX memory_stats | Memory usage | Device-level | Python dict |

---

## 1. NSight Systems (nsys)

**Best for:** Overall timeline, kernel launches, CPU-GPU synchronization, identifying bottlenecks.

### Basic Usage

```bash
# Profile entire run
nsys profile -o my_profile python3 my_script.py --device gpu

# With CUDA and NVTX annotations
nsys profile -t cuda,nvtx -o my_profile python3 my_script.py

# Capture GPU metrics
nsys profile --gpu-metrics-device=0 -o my_profile python3 my_script.py
```

### View Results

```bash
# GUI (requires X11 forwarding or local install)
nsys-ui my_profile.nsys-rep

# CLI stats summary
nsys stats my_profile.nsys-rep
```

### Key Metrics to Look For

- **CUDA API calls** — Time in cudaLaunchKernel, cudaMemcpy, cudaStreamSynchronize
- **Kernel duration** — Are kernels actually running long, or is launch overhead dominating?
- **Gaps in timeline** — CPU-bound sections, synchronization stalls
- **Memory transfers** — Host↔Device copies (should be minimal after setup)

### Example Analysis

```bash
# Generate summary report
nsys stats --report cuda_gpu_kern_sum my_profile.nsys-rep

# Output shows kernel execution times:
# Time(%)  Total Time  Instances  Avg        Name
# 45.2%    12.34 ms    100        123.4 us   xla_computation_gradient
# 32.1%    8.76 ms     100        87.6 us    xla_computation_divergence
```

---

## 2. NSight Compute (ncu)

**Best for:** Deep kernel analysis — occupancy, memory throughput, compute utilization.

### Basic Usage

```bash
# Profile all kernels (can be slow)
ncu -o kernel_profile python3 my_script.py

# Profile specific kernel pattern
ncu --kernel-name "regex:xla.*gradient" -o kernel_profile python3 my_script.py

# Full metrics set
ncu --set full -o kernel_profile python3 my_script.py

# Quick roofline analysis
ncu --set roofline -o kernel_profile python3 my_script.py
```

### Key Metrics

| Metric | What It Tells You |
|--------|-------------------|
| **SM Occupancy** | % of GPU threads active (target: >50%) |
| **Memory Throughput** | GB/s achieved vs peak |
| **Compute Throughput** | FLOPS achieved vs peak |
| **L1/L2 Hit Rate** | Cache efficiency |
| **Memory Bound vs Compute Bound** | Roofline position |

### Roofline Analysis

```bash
ncu --set roofline --target-processes all -o roofline python3 my_script.py
```

This generates a roofline plot showing whether kernels are:
- **Memory-bound** (below the roofline) — optimize memory access patterns
- **Compute-bound** (at the roofline) — optimize arithmetic

For our SWE solver, expect to be **memory-bound** (0.5-2% compute utilization typical for stencil codes).

---

## 3. JAX Built-in Profiler

**Best for:** Understanding JAX/XLA compilation, operation fusion, HLO graph.

### Chrome Trace

```python
import jax

# Start profiling
jax.profiler.start_trace("/tmp/jax-trace")

# Your computation
for i in range(10):
    result = my_jit_function(data)
    jax.block_until_ready(result)

# Stop and write trace
jax.profiler.stop_trace()
```

View in Chrome:
1. Open `chrome://tracing`
2. Load the trace file from `/tmp/jax-trace/`

### TensorBoard Integration

```python
# Requires: pip install tensorboard-plugin-profile
jax.profiler.start_trace("/tmp/tensorboard-logs")
# ... computation ...
jax.profiler.stop_trace()
```

```bash
tensorboard --logdir=/tmp/tensorboard-logs
# Open http://localhost:6006 and navigate to Profile tab
```

### What to Look For

- **XLA compilation time** — Should happen once during warmup
- **Operation fusion** — XLA should fuse elementwise ops
- **Memory allocation** — Excessive allocations indicate issues
- **Device execution** — Actual GPU kernel time

---

## 4. JAX Memory Statistics

**Best for:** Quick memory usage check without external tools.

```python
import jax
import jax.numpy as jnp

def print_memory_stats():
    """Print GPU memory statistics."""
    for device in jax.devices():
        if device.platform == 'gpu':
            stats = device.memory_stats()
            if stats:
                print(f"\nDevice: {device}")
                print(f"  Bytes in use:     {stats.get('bytes_in_use', 0) / 1e9:.3f} GB")
                print(f"  Peak bytes:       {stats.get('peak_bytes_in_use', 0) / 1e9:.3f} GB")
                print(f"  Bytes reserved:   {stats.get('bytes_reserved', 0) / 1e9:.3f} GB")
                print(f"  Bytes limit:      {stats.get('bytes_limit', 0) / 1e9:.3f} GB")

# Usage
h = jnp.zeros((6, 1200, 1200))  # Allocate
print_memory_stats()
```

### Memory Estimation for SWE Solver

For N=1200, 6 faces, FP32:

| Array | Shape | Size |
|-------|-------|------|
| h | (6, 1200, 1200) | 33 MB |
| Vx, Vy, Vz | 3 × (6, 1200, 1200) | 99 MB |
| B (Bernoulli) | (6, 1200, 1200) | 33 MB |
| Jacobians (6 components) | 6 × (1200, 1200) | 33 MB |
| sqrtG | (6, 1200, 1200) | 33 MB |
| Temporaries | ~5× state | 165 MB |
| **Total estimate** | | **~400-600 MB** |

A10G has 24 GB — plenty of headroom for N=1200.

---

## 5. Practical Profiling Workflow

### Step 1: Quick Sanity Check

```python
# Add to your benchmark script
print_memory_stats()
```

### Step 2: Timeline Overview

```bash
nsys profile -o timeline python3 my_script.py --benchmark --N 600
nsys stats timeline.nsys-rep
```

Look for:
- Is time spent in kernels or overhead?
- Any unexpected CPU-GPU synchronization?

### Step 3: Identify Hot Kernels

```bash
nsys stats --report cuda_gpu_kern_sum timeline.nsys-rep | head -20
```

### Step 4: Deep Dive on Hot Kernels

```bash
ncu --kernel-name "regex:hot_kernel_name" --set roofline -o deep_dive python3 my_script.py
```

### Step 5: JAX-Level Analysis (if needed)

```python
jax.profiler.start_trace("/tmp/jax-trace")
# ... run hot path ...
jax.profiler.stop_trace()
```

---

## 6. Common Issues and Solutions

### Issue: High JIT Compilation Time

**Symptom:** First iteration is 100-1000x slower than subsequent ones.

**Solution:** Already handled — use warmup iterations before timing.

### Issue: Excessive Memory Allocation

**Symptom:** Memory usage grows over iterations.

**Solution:** 
```python
# Force garbage collection
jax.clear_caches()

# Pre-allocate output arrays
output = jnp.zeros_like(input)
```

### Issue: Low GPU Utilization

**Symptom:** NSight shows gaps in kernel execution.

**Solution:**
- Batch operations to amortize kernel launch overhead
- Use `jax.lax.scan` instead of Python loops
- Ensure `jax.block_until_ready()` only at measurement boundaries

### Issue: Memory-Bound Kernels

**Symptom:** NCU roofline shows kernels far below compute ceiling.

**Solution:** (for stencil codes, this is expected)
- Fuse operations to reduce memory traffic
- Use shared memory / cache blocking (requires custom kernels)
- Ensure coalesced memory access patterns

---

## 7. Quick Reference

```bash
# NSight Systems - timeline
nsys profile -o prof python3 script.py
nsys stats prof.nsys-rep

# NSight Compute - kernel details  
ncu --set roofline -o prof python3 script.py

# JAX trace
python3 -c "
import jax
jax.profiler.start_trace('/tmp/trace')
# ... your code ...
jax.profiler.stop_trace()
"

# Memory check
python3 -c "
import jax
for d in jax.devices():
    print(d, d.memory_stats())
"
```

---

## References

- [NSight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [NSight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- [JAX Profiling Documentation](https://jax.readthedocs.io/en/latest/profiling.html)
- [XLA Tools](https://www.tensorflow.org/xla/tools)
