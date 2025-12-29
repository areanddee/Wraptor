# GPU Testing Guide

## Prerequisites

1. **Install Wraptor** (follow main README.md)
2. **Install JAX with CUDA support:**
   ```bash
   pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

## Quick GPU Verification

### Test 1: Check GPU Visibility

```bash
# Should show your H100s
nvidia-smi

# Should show CudaDevice(id=0) etc.
python -c "import jax; print(jax.devices())"
```

### Test 2: Simple GPU Test (3 minutes)

```bash
# Make sure environment is activated
source wraptor_env/bin/activate

# Set JAX to use GPU
export JAX_PLATFORMS=cuda

# Run simple test
python Tests/test_gpu_simple.py
```

**Expected output:**
```
✅ JAX is using GPU backend: cuda
✅ Computation correct
✅ Diffusion solver working
🎉 All GPU tests PASSED!
```

---

## SLURM Batch Job (Automated)

### Edit SLURM Script

```bash
# Edit gpu_test_diffusion.sh
# Change line 3: #SBATCH --account=YOUR_ACCOUNT
# Change line 4: #SBATCH --partition=gpu  (use your GPU partition name)
```

### Submit Job

```bash
# Make executable
chmod +x gpu_test_diffusion.sh

# Submit
sbatch gpu_test_diffusion.sh

# Check status
squeue -u $USER

# View output when done
cat gpu_test_*.out
```

---

## Interactive GPU Session (Development)

### Option 1: salloc (Recommended for first time)

```bash
# Request interactive GPU node (adjust partition/account)
salloc --gres=gpu:1 --time=01:00:00 --partition=gpu --account=YOUR_ACCOUNT

# Once allocated, you'll get a shell on the GPU node
# Activate environment
source wraptor_env/bin/activate

# Set GPU mode
export JAX_PLATFORMS=cuda

# Verify
python -c "import jax; print(jax.devices())"

# Run tests interactively
python Tests/test_gpu_simple.py

# Exit when done
exit
```

### Option 2: sbatch --interactive

```bash
# Edit gpu_interactive.sh first (set account/partition)
sbatch --interactive gpu_interactive.sh
```

---

## Troubleshooting

### "JAX is using CPU, not GPU"

```bash
# Make sure CUDA is visible
nvidia-smi

# Set JAX platform explicitly
export JAX_PLATFORMS=cuda

# Verify
python -c "import jax; print(jax.default_backend())"
# Should say: cuda
```

### "CUDA not found" or "libcuda.so not found"

```bash
# Load CUDA module
module load cuda/12.x

# Or set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### JAX installed but GPU version not found

```bash
# Reinstall JAX with CUDA
pip uninstall jax jaxlib
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### "Out of memory" on GPU

```bash
# Reduce resolution in config
# Edit Config/diffusion_framework.yaml
# Change N: 120 → N: 30
```

---

## Performance Expectations

### H100 (80GB) - Diffusion Solver

| Resolution | Steps/sec (CPU) | Steps/sec (1xH100) | Speedup |
|------------|-----------------|---------------------|---------|
| N=30 | ~50 | ~500 | 10x |
| N=60 | ~10 | ~150 | 15x |
| N=120 | ~2 | ~40 | 20x |

**Note:** First run will be slower due to JIT compilation. Subsequent runs are faster.

---

## Multi-GPU Testing (Advanced)

Once single GPU works, you can test with multiple H100s:

```bash
# Request 4 GPUs
salloc --gres=gpu:4 --time=01:00:00

# Enable sharding in config
# Edit Config/diffusion_framework.yaml:
parallelization:
  enable_sharding: true
  num_devices: 4
  device_type: 'gpu'

# Run
python -m Framework.runner Config/diffusion_framework.yaml
```

---

## Next Steps

1. ✅ Verify single GPU works (`test_gpu_simple.py`)
2. Run full test suite on GPU (`Tests/run_all_tests.py`)
3. Test multi-GPU sharding (2, 4, 8 GPUs)
4. Benchmark performance vs CPU
5. Test larger problems (N=240, N=480)



