// Implicit3D.cpp
//
// Sequential 3D heat equation solver using the Implicit (Backward Euler) scheme,
// with the resulting linear system solved by Jacobi iteration at every time step.
// Direct 3D extension of heat2d_implicit.cpp -- same idea, one more dimension.
//
// Solves:   du/dt = alpha * (d2u/dx2 + d2u/dy2 + d2u/dz2)   on the unit cube [0,1]^3
// with homogeneous Dirichlet boundary conditions (u = 0 on all six faces).
//
// Verified against the same 3D exact manufactured solution used for heat3d_explicit.cpp:
//
//     u_exact(x,y,z,t) = sin(pi*x) * sin(pi*y) * sin(pi*z) * exp(-3 * alpha * pi^2 * t)

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

const double PI = 3.14159265358979323846;

double exact_solution(double x, double y, double z, double t, double alpha) {
    return std::sin(PI * x) * std::sin(PI * y) * std::sin(PI * z)
           * std::exp(-3.0 * alpha * PI * PI * t);
}

int main() {
    // ---------------- Problem parameters ----------------
    const double alpha   = 1.0;
    const int    N       = 20;   // same resolution as heat3d_explicit.cpp
    const double L       = 1.0;
    const double h       = L / (N + 1);
    const double t_final = 0.02;

    // No stability limit for implicit -- pick a time step well above the explicit
    // 3D limit of r <= 1/6, to demonstrate the same advantage shown in 2D.
    const double r         = 1.0;
    const double dt         = r * h * h / alpha;
    const int    num_steps  = static_cast<int>(t_final / dt);

    // ---------------- Jacobi iteration settings ----------------
    const int    max_jacobi_iters = 10000;
    const double jacobi_tolerance = 1e-10;

    std::cout << "=== 3D Implicit Heat Equation Solver (Backward Euler + Jacobi) ===\n";
    std::cout << "Grid spacing h       = " << h << "\n";
    std::cout << "Time step dt         = " << dt << "\n";
    std::cout << "Stability number r   = " << r << "  (no upper limit needed here)\n";
    std::cout << "Number of time steps = " << num_steps << "\n";
    std::cout << "Total grid points    = " << (long)(N+2)*(N+2)*(N+2) << "\n\n";

    // ---------------- Storage ----------------
    int size = N + 2;
    std::vector<std::vector<std::vector<double>>> u(
        size, std::vector<std::vector<double>>(size, std::vector<double>(size, 0.0)));
    std::vector<std::vector<std::vector<double>>> u_old = u;
    std::vector<std::vector<std::vector<double>>> u_new = u;

    // ---------------- Initial condition ----------------
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            for (int k = 0; k < size; ++k) {
                double x = i * h, y = j * h, z = k * h;
                u[i][j][k] = exact_solution(x, y, z, 0.0, alpha);
            }

    // ---------------- Time-stepping loop ----------------
    for (int step = 0; step < num_steps; ++step) {
        u_old = u;   // this step's fixed starting values
        u_new = u;   // Jacobi's first guess

        // ---- Jacobi iteration: guess, check, improve, repeat ----
        for (int iter = 0; iter < max_jacobi_iters; ++iter) {
            double max_change = 0.0;
            std::vector<std::vector<std::vector<double>>> u_guess = u_new;

            for (int i = 1; i <= N; ++i)
                for (int j = 1; j <= N; ++j)
                    for (int k = 1; k <= N; ++k) {
                        // Backward Euler, rearranged to solve for u_new(i,j,k):
                        //   u_new = [u_old + r*(6 neighbours, from LAST guess)] / (1 + 6r)
                        double neighbour_sum = u_guess[i+1][j][k] + u_guess[i-1][j][k]
                                              + u_guess[i][j+1][k] + u_guess[i][j-1][k]
                                              + u_guess[i][j][k+1] + u_guess[i][j][k-1];
                        double updated = (u_old[i][j][k] + r * neighbour_sum) / (1.0 + 6.0 * r);
                        max_change = std::max(max_change, std::abs(updated - u_guess[i][j][k]));
                        u_new[i][j][k] = updated;
                    }
            // Boundaries stay at 0 throughout -- never touched, matches Dirichlet BC.

            if (max_change < jacobi_tolerance) {
                break;
            }
        }

        u = u_new;
    }

    // ---------------- Verification against the exact solution ----------------
    double actual_t_final = num_steps * dt;
    double max_error = 0.0, l2_sum = 0.0;
    long count = 0;
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            for (int k = 0; k < size; ++k) {
                double x = i * h, y = j * h, z = k * h;
                double exact = exact_solution(x, y, z, actual_t_final, alpha);
                double error = std::abs(u[i][j][k] - exact);
                max_error = std::max(max_error, error);
                l2_sum += error * error;
                count++;
            }
    double l2_error = std::sqrt(l2_sum / count);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Simulated up to t    = " << actual_t_final << "\n";
    std::cout << "Max error vs exact   = " << max_error << "\n";
    std::cout << "L2 error vs exact    = " << l2_error << "\n";

    return 0;
}