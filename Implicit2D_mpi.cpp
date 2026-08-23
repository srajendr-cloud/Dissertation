#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cstdlib>

const double PI = 3.14159265358979323846;

double exact_solution(double x, double y, double t, double alpha) {
    return std::sin(PI * x) * std::sin(PI * y) * std::exp(-2.0 * alpha * PI * PI * t);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const double alpha    = 1.0;
    // N can be set from the command line for weak scaling experiments; defaults to
    // 40
    const int    N        = (argc > 1) ? std::atoi(argv[1]) : 40;
    const double L        = 1.0;
    const double h        = L / (N + 1);
    const double t_final  = 0.05;
    const double r        = 1.0; // no stability limit needed for implicit
    const double dt        = r * h * h / alpha;
    const int    num_steps = static_cast<int>(t_final / dt);

    const int    max_jacobi_iters = 10000;
    const double jacobi_tolerance = 1e-10;

    int base_rows  = N / num_procs;
    int remainder  = N % num_procs;
    int local_rows = base_rows + (rank < remainder ? 1 : 0);
    int row_start  = (rank < remainder)
                      ? rank * (base_rows + 1) + 1
                      : remainder * (base_rows + 1) + (rank - remainder) * base_rows + 1;

    if (rank == 0) {
        std::cout << "=== 2D Implicit Heat Equation Solver (MPI, Backward Euler + Jacobi) ===\n";
        std::cout << "Running on " << num_procs << " process(es)\n";
        std::cout << "Grid spacing h       = " << h << "\n";
        std::cout << "Time step dt         = " << dt << "\n";
        std::cout << "Number of time steps = " << num_steps << "\n\n";
    }

    int cols = N + 2;
    int local_size = local_rows + 2;
    std::vector<std::vector<double>> u(local_size, std::vector<double>(cols, 0.0));
    std::vector<std::vector<double>> u_old(local_size, std::vector<double>(cols, 0.0));
    std::vector<std::vector<double>> u_new(local_size, std::vector<double>(cols, 0.0));

    for (int li = 0; li <= local_rows + 1; ++li) {
        int global_i = row_start + (li - 1);
        if (global_i < 0 || global_i > N + 1) continue;
        for (int j = 0; j < cols; ++j) {
            double x = global_i * h;
            double y = j * h;
            u[li][j] = exact_solution(x, y, 0.0, alpha);
        }
    }

    bool has_upper_neighbour = (rank > 0);
    bool has_lower_neighbour = (rank < num_procs - 1);

    // Small helper: exchange halo rows for whichever array is passed in.
    auto exchange_halos = [&](std::vector<std::vector<double>>& arr) {
        if (has_upper_neighbour) {
            MPI_Sendrecv(arr[1].data(), cols, MPI_DOUBLE, rank - 1, 0,
                         arr[0].data(), cols, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (has_lower_neighbour) {
            MPI_Sendrecv(arr[local_rows].data(),     cols, MPI_DOUBLE, rank + 1, 0,
                         arr[local_rows + 1].data(), cols, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    };

    // ---------------- Timing: start the clock, synchronised across all processes ----------------
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int step = 0; step < num_steps; ++step) {
        u_old = u;
        u_new = u;
        exchange_halos(u_old); // neighbours' fixed starting values for this whole step

        for (int iter = 0; iter < max_jacobi_iters; ++iter) {
            exchange_halos(u_new); // neighbours' LATEST guess -- needed every iteration

            double local_max_change = 0.0;
            std::vector<std::vector<double>> u_guess = u_new;

            for (int li = 1; li <= local_rows; ++li) {
                int global_i = row_start + (li - 1);
                if (global_i == 0 || global_i == N + 1) continue;
                for (int j = 1; j <= N; ++j) {
                    double neighbour_sum = u_guess[li+1][j] + u_guess[li-1][j]
                                          + u_guess[li][j+1] + u_guess[li][j-1];
                    double updated = (u_old[li][j] + r * neighbour_sum) / (1.0 + 4.0 * r);
                    local_max_change = std::max(local_max_change, std::abs(updated - u_guess[li][j]));
                    u_new[li][j] = updated;
                }
            }

            // Every process must agree the WHOLE grid has settled, not just its own patch.
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

    // ---------------- Verification ----------------
    double actual_t_final = num_steps * dt;
    double local_max_error = 0.0, local_l2_sum = 0.0;
    for (int li = 1; li <= local_rows; ++li) {
        int global_i = row_start + (li - 1);
        for (int j = 0; j < cols; ++j) {
            double x = global_i * h;
            double y = j * h;
            double exact = exact_solution(x, y, actual_t_final, alpha);
            double error = std::abs(u[li][j] - exact);
            local_max_error = std::max(local_max_error, error);
            local_l2_sum += error * error;
        }
    }

    double global_max_error = 0.0, global_l2_sum = 0.0;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_l2_sum,    &global_l2_sum,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double global_l2_error = std::sqrt(global_l2_sum / ((N + 2) * (N + 2)));
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Simulated up to t    = " << actual_t_final << "\n";
        std::cout << "Max error vs exact   = " << global_max_error << "\n";
        std::cout << "L2 error vs exact    = " << global_l2_error << "\n";
        std::cout << "Elapsed time (s)     = " << max_elapsed << "\n";
    }

    MPI_Finalize();
    return 0;
}
