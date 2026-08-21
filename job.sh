#!/bin/bash
#SBATCH --job-name=heat_scaling
#SBATCH --output=heat_scaling_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

module load openmpi
export LD_LIBRARY_PATH=/home/support/rl8/spack/1.1.1/opt/spack/linux-x86_64_v3/openmpi-5.0.9-2irqibqfnnap6fptp26b7bnwf5qhybuq/lib:$LD_LIBRARY_PATH

for np in 1 2 4 8 16 32
do
    echo "===== Explicit2D_mpi with $np processes ====="
    mpirun -np $np ./Explicit2D_mpi
    echo ""
done

for np in 1 2 4 8 16 32
do
    echo "===== Explicit3D_mpi with $np processes ====="
    mpirun -np $np ./Explicit3D_mpi
    echo ""
done

for np in 1 2 4 8 16 32
do
    echo "===== Implicit2D_mpi with $np processes ====="
    mpirun -np $np ./Implicit2D_mpi
    echo ""
done

for np in 1 2 4 8 16 32
do
    echo "===== Implicit3D_mpi with $np processes ====="
    mpirun -np $np ./Implicit3D_mpi
    echo ""
done
