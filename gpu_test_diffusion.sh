#!/bin/bash
#SBATCH --job-name=wraptor_gpu_test
#SBATCH --account=YOUR_ACCOUNT      # Change this
#SBATCH --partition=gpu             # Change to your GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=00:10:00             # 10 minutes should be plenty
#SBATCH --output=gpu_test_%j.out
#SBATCH --error=gpu_test_%j.err

# Load modules if needed
# module load cuda/12.x
# module load python/3.9

# Activate virtual environment
source wraptor_env/bin/activate

# Set JAX to use GPU
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0

# Print GPU info
echo "==================================================================="
echo "GPU TEST - Diffusion Solver"
echo "==================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Verify JAX sees GPU
echo "JAX device detection:"
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices()); print('Default backend:', jax.default_backend())"
echo ""

# Run single diffusion test
echo "Running diffusion solver test..."
python Tests/test_solvers.py::test_diffusion_heat_conservation -v

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GPU test PASSED"
else
    echo ""
    echo "❌ GPU test FAILED"
    exit 1
fi

# Monitor GPU usage during run (optional)
# nvidia-smi dmon -s u -c 10 &



