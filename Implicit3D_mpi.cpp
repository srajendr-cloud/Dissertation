#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cstdlib>

const double PI = 3.14159265358979323846;

double exact_solution(double x, double y, double z, double t, double alpha) {
    return std::sin(PI * x) * std::sin(PI * y) * std::sin(PI * z)
           * std::exp(-3.0 * alpha * PI * PI * t);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const double alpha   = 1.0;
    // N can be set from the command line for weak scaling experiments; defaults to
    // 20, matching the strong scaling baseline.
    const int    N       = (argc > 1) ? std::atoi(argv[1]) : 20;
    const double L       = 1.0;
    const double h       = L / (N + 1);
    const double t_final = 0.02;
    const double r        = 1.0;
    const double dt        = r * h * h / alpha;
    const int    num_steps = static_cast<int>(t_final / dt);

    const int    max_jacobi_iters = 10000;
    const double jacobi_tolerance = 1e-10;

    int base_slabs = N / num_procs;
    int remainder   = N % num_procs;
    int local_i     = base_slabs + (rank < remainder ? 1 : 0);
    int i_start     = (rank < remainder)
                       ? rank * (base_slabs + 1) + 1
                       : remainder * (base_slabs + 1) + (rank - remainder) * base_slabs + 1;

    if (rank == 0) {
        std::cout << "=== 3D Implicit Heat Equation Solver (MPI, Backward Euler + Jacobi) ===\n";
        std::cout << "Running on " << num_procs << " process(es)\n";
        std::cout << "Grid spacing h       = " << h << "\n";
        std::cout << "Time step dt         = " << dt << "\n";
        std::cout << "Number of time steps = " << num_steps << "\n\n";
    }

    int plane = (N + 2) * (N + 2);
    int local_size_i = local_i + 2;
    std::vector<double> u(local_size_i * plane, 0.0);
    std::vector<double> u_old(local_size_i * plane, 0.0);
    std::vector<double> u_new(local_size_i * plane, 0.0);

    auto idx = [&](int li, int j, int k) {
        return li * plane + j * (N + 2) + k;
    };

    for (int li = 0; li <= local_i + 1; ++li) {
        int global_i = i_start + (li - 1);
        if (global_i < 0 || global_i > N + 1) continue;
        for (int j = 0; j < N + 2; ++j)
            for (int k = 0; k < N + 2; ++k) {
                double x = global_i * h, y = j * h, z = k * h;
                u[idx(li, j, k)] = exact_solution(x, y, z, 0.0, alpha);
            }
    }

    bool has_upper_neighbour = (rank > 0);
    bool has_lower_neighbour = (rank < num_procs - 1);

    auto exchange_halos = [&](std::vector<double>& arr) {
        if (has_upper_neighbour) {
            MPI_Sendrecv(&arr[idx(1, 0, 0)], plane, MPI_DOUBLE, rank - 1, 0,
                         &arr[idx(0, 0, 0)], plane, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (has_lower_neighbour) {
            MPI_Sendrecv(&arr[idx(local_i, 0, 0)],     plane, MPI_DOUBLE, rank + 1, 0,
                         &arr[idx(local_i + 1, 0, 0)], plane, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    };

    // ---------------- Timing: start the clock, synchronised across all processes ----------------
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int step = 0; step < num_steps; ++step) {
        u_old = u;
        u_new = u;
        exchange_halos(u_old);

        for (int iter = 0; iter < max_jacobi_iters; ++iter) {
            exchange_halos(u_new);

            double local_max_change = 0.0;
            std::vector<double> u_guess = u_new;

            for (int li = 1; li <= local_i; ++li) {
                int global_i = i_start + (li - 1);
                if (global_i == 0 || global_i == N + 1) continue;
                for (int j = 1; j <= N; ++j)
                    for (int k = 1; k <= N; ++k) {
                        double neighbour_sum = u_guess[idx(li+1,j,k)] + u_guess[idx(li-1,j,k)]
                                              + u_guess[idx(li,j+1,k)] + u_guess[idx(li,j-1,k)]
                                              + u_guess[idx(li,j,k+1)] + u_guess[idx(li,j,k-1)];
                        double updated = (u_old[idx(li,j,k)] + r * neighbour_sum) / (1.0 + 6.0 * r);
                        local_max_change = std::max(local_max_change, std::abs(updated - u_guess[idx(li,j,k)]));
                        u_new[idx(li,j,k)] = updated;
                    }
            }

            double global_max_change = 0.0;
            MPI_Allreduce(&local_max_change, &global_max_change, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (global_max_change < jacobi_tolerance) break;
        }

        u = u_new;
    }

    // ---------------- Timing: stop the clock, take the slowest process's time ----------------
    double local_elapsed = MPI_Wtime() - start_time;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double actual_t_final = num_steps * dt;
    double local_max_error = 0.0, local_l2_sum = 0.0;
    for (int li = 1; li <= local_i; ++li) {
        int global_i = i_start + (li - 1);
        for (int j = 0; j < N + 2; ++j)
            for (int k = 0; k < N + 2; ++k) {
                double x = global_i * h, y = j * h, z = k * h;
                double exact = exact_solution(x, y, z, actual_t_final, alpha);
                double error = std::abs(u[idx(li,j,k)] - exact);
                local_max_error = std::max(local_max_error, error);
                local_l2_sum += error * error;
            }
    }

    double global_max_error = 0.0, global_l2_sum = 0.0;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_l2_sum,    &global_l2_sum,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long total_points = (long)(N + 2) * (N + 2) * (N + 2);
        double global_l2_error = std::sqrt(global_l2_sum / total_points);
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Simulated up to t    = " << actual_t_final << "\n";
        std::cout << "Max error vs exact   = " << global_max_error << "\n";
        std::cout << "L2 error vs exact    = " << global_l2_error << "\n";
        std::cout << "Elapsed time (s)     = " << max_elapsed << "\n";
    }

    MPI_Finalize();
    return 0;
}
