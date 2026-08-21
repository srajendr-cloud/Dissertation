#!/bin/bash
#SBATCH --job-name=heat_weak_scaling
#SBATCH --output=heat_weak_scaling_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

module load openmpi
export LD_LIBRARY_PATH=/home/support/rl8/spack/1.1.1/opt/spack/linux-x86_64_v3/openmpi-5.0.9-2irqibqfnnap6fptp26b7bnwf5qhybuq/lib:$LD_LIBRARY_PATH

np_2d=(1 2 4 8 16 32)
n_2d=(40 57 80 113 160 226)

for i in 0 1 2 3 4 5
do
    np=${np_2d[$i]}
    n=${n_2d[$i]}
    echo "===== Explicit2D_mpi weak scaling: $np processes, N=$n ====="
    mpirun -np $np ./Explicit2D_mpi $n
    echo ""
done

for i in 0 1 2 3 4 5
do
    np=${np_2d[$i]}
    n=${n_2d[$i]}
    echo "===== Implicit2D_mpi weak scaling: $np processes, N=$n ====="
    mpirun -np $np ./Implicit2D_mpi $n
    echo ""
done

np_3d=(1 2 4 8 16 32)
n_3d=(20 25 32 40 50 63)

for i in 0 1 2 3 4 5
do
    np=${np_3d[$i]}
    n=${n_3d[$i]}
    echo "===== Explicit3D_mpi weak scaling: $np processes, N=$n ====="
    mpirun -np $np ./Explicit3D_mpi $n
    echo ""
done

for i in 0 1 2 3 4 5
do
    np=${np_3d[$i]}
    n=${n_3d[$i]}
    echo "===== Implicit3D_mpi weak scaling: $np processes, N=$n ====="
    mpirun -np $np ./Implicit3D_mpi $n
    echo ""
done
