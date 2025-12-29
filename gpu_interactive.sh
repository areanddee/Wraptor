#!/bin/bash
# Interactive GPU session for development/debugging
# Usage: sbatch --interactive gpu_interactive.sh

#SBATCH --job-name=wraptor_interactive
#SBATCH --account=YOUR_ACCOUNT      # Change this
#SBATCH --partition=gpu             # Change to your GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                # 1 GPU for testing
#SBATCH --time=01:00:00             # 1 hour interactive session
#SBATCH --output=interactive_%j.out

# Or just use: salloc --gres=gpu:1 --time=01:00:00 --partition=gpu

echo "==================================================================="
echo "Interactive GPU Session"
echo "==================================================================="
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo ""
echo "To activate environment: source wraptor_env/bin/activate"
echo "To test GPU: python -c 'import jax; print(jax.devices())'"
echo "==================================================================="

# Keep session alive
bash

