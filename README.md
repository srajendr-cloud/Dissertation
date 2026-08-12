# Performance and Scaling Analysis of Parallel Discretisations for the Heat Equation in 2D and 3D

MSc dissertation code by Sreelakshmi Rajendrakumar, Trinity College Dublin.

This repository holds the code and results for a dissertation comparing explicit and implicit finite difference methods for the heat equation, in 2D and 3D, run sequentially and in parallel using MPI.

## Files

**Sequential solvers**
- Explicit2D.cpp, Explicit3D.cpp - explicit (Forward Euler) method
- Implicit2D.cpp, Implicit3D.cpp - implicit (Backward Euler) method, solved using Jacobi iteration

**MPI parallel solvers**
- Explicit2D_mpi.cpp, Explicit3D_mpi.cpp
- Implicit2D_mpi.cpp, Implicit3D_mpi.cpp

**Cluster job scripts**
- job.sh - strong scaling experiment
- job_weak.sh - weak scaling experiment

**Results**
- heat_scaling_47241.out - strong scaling results
- heat_weak_scaling_47242.out - weak scaling results

## How to compile and run

Sequential:
g++ -O2 -std=c++17 -o Explicit2D Explicit2D.cpp
./Explicit2D

MPI:
mpic++ -O2 -std=c++17 -o Explicit2D_mpi Explicit2D_mpi.cpp
mpirun -np 4 ./Explicit2D_mpi

