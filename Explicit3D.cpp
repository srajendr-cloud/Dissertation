// Explicit3D.cpp
//
// Sequential 3D heat equation solver using the Explicit (Forward Euler / FTCS) scheme.
// This is the direct 3D extension of heat2d_explicit.cpp -- same idea, one more dimension.
//
// Solves:   du/dt = alpha * (d2u/dx2 + d2u/dy2 + d2u/dz2)   on the unit cube [0,1]^3
// with homogeneous Dirichlet boundary conditions (u = 0 on all six faces).
//
// Verified against the exact manufactured solution:
//
//     u_exact(x,y,z,t) = sin(pi*x) * sin(pi*y) * sin(pi*z) * exp(-3 * alpha * pi^2 * t)
//
// Note the exponent here is -3*alpha*pi^2*t (one pi^2 term per dimension), compared
// to -2*alpha*pi^2*t in the 2D version -- this is the direct mathematical reason heat
// diffuses away faster, in relative terms, as you add spatial dimensions.

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
    const int    N       = 20;   // interior points per direction -- 3D grids get big fast:
                                  // N=20 here already means (N+2)^3 = 10,648 points.
    const double L       = 1.0;
    const double h       = L / (N + 1);
    const double t_final = 0.02;

    // Stability condition for 3D explicit FTCS: r = alpha*dt/h^2 must be <= 1/6.
    // This is TIGHTER than the 2D limit of 1/4 -- one extra neighbour direction
    // means less "safety margin" per step, so 3D forces smaller time steps for the
    // same spatial resolution. This is a genuinely important, reportable difference
    // between 2D and 3D scaling, independent of how many grid points you use.
    const double r        = 0.15;
    const double dt        = r * h * h / alpha;
    const int    num_steps = static_cast<int>(t_final / dt);

    std::cout << "=== 3D Explicit Heat Equation Solver ===\n";
    std::cout << "Grid spacing h       = " << h << "\n";
    std::cout << "Time step dt         = " << dt << "\n";
    std::cout << "Stability number r   = " << r << "  (must be <= 0.1667 in 3D)\n";
    std::cout << "Number of time steps = " << num_steps << "\n";
    std::cout << "Total grid points    = " << (long)(N+2)*(N+2)*(N+2) << "\n\n";

    // ---------------- Storage ----------------
    int size = N + 2;
    std::vector<std::vector<std::vector<double>>> u(
        size, std::vector<std::vector<double>>(size, std::vector<double>(size, 0.0)));
    std::vector<std::vector<std::vector<double>>> u_new = u;

    // ---------------- Initial condition ----------------
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            for (int k = 0; k < size; ++k) {
                double x = i * h, y = j * h, z = k * h;
                u[i][j][k] = exact_solution(x, y, z, 0.0, alpha);
            }
    u_new = u; // boundary faces start at 0, exactly as in the 2D version

    // ---------------- Time-stepping loop ----------------
    for (int step = 0; step < num_steps; ++step) {
        for (int i = 1; i <= N; ++i)
            for (int j = 1; j <= N; ++j)
                for (int k = 1; k <= N; ++k) {
                    double laplacian = (u[i+1][j][k] + u[i-1][j][k]
                                       + u[i][j+1][k] + u[i][j-1][k]
                                       + u[i][j][k+1] + u[i][j][k-1]
                                       - 6.0 * u[i][j][k]) / (h * h);
                    u_new[i][j][k] = u[i][j][k] + alpha * dt * laplacian;
                }
        std::swap(u, u_new); // boundary faces of u_new were never touched, stay 0
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