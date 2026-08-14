// Explicit2D.cpp
//
// Sequential 2D heat equation solver using the Explicit (Forward Euler / FTCS) scheme.
//
// Solves:   du/dt = alpha * (d2u/dx2 + d2u/dy2)   on the unit square [0,1] x [0,1]
// with homogeneous Dirichlet boundary conditions (u = 0 on all four edges).
//
// We verify the code is correct by comparing against an EXACT manufactured solution:
//
//     u_exact(x, y, t) = sin(pi*x) * sin(pi*y) * exp(-2 * alpha * pi^2 * t)
//
// This function satisfies the heat equation exactly (you can check this with calculus),
// and it is exactly 0 on the boundary (since sin(0) = sin(pi) = 0), so it fits our
// boundary conditions perfectly. This lets us compute a real numerical error, not just
// "the code ran without crashing".

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

const double PI = 3.14159265358979323846;

// The exact solution. Used both to set the initial condition (at t=0) and later
// to check our numerical answer against the true answer (at t=t_final).
double exact_solution(double x, double y, double t, double alpha) {
    return std::sin(PI * x) * std::sin(PI * y) * std::exp(-2.0 * alpha * PI * PI * t);
}

int main() {
    // ---------------- Problem parameters ----------------
    const double alpha    = 1.0;   // thermal diffusivity
    const int    N        = 40;    // number of INTERIOR grid points in each direction
    const double L        = 1.0;   // domain size: the unit square
    const double h        = L / (N + 1); // grid spacing
    const double t_final  = 0.05;  // how far in time we simulate

    // Stability condition for 2D explicit FTCS:  r = alpha*dt/h^2  must be <= 1/4.
    // We deliberately choose r comfortably below that limit.
    const double r        = 0.20;
    const double dt        = r * h * h / alpha;
    const int    num_steps = static_cast<int>(t_final / dt);

    std::cout << "=== 2D Explicit Heat Equation Solver ===\n";
    std::cout << "Grid spacing h       = " << h << "\n";
    std::cout << "Time step dt         = " << dt << "\n";
    std::cout << "Stability number r   = " << r << "  (must be <= 0.25 in 2D)\n";
    std::cout << "Number of time steps = " << num_steps << "\n\n";

    // ---------------- Storage ----------------
    // size = N+2 so that index i runs 0..N+1, with i=0 and i=N+1 being the two
    // boundary edges, and i=1..N being the interior points we actually solve for.
    int size = N + 2;
    std::vector<std::vector<double>> u(size, std::vector<double>(size, 0.0));
    std::vector<std::vector<double>> u_new(size, std::vector<double>(size, 0.0));

    // ---------------- Initial condition ----------------
    // Set u(x, y, 0) from the exact solution at t=0. This also automatically gives
    // us u=0 on the boundary, since sin(pi*0) = sin(pi*1) = 0.
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double x = i * h;
            double y = j * h;
            u[i][j] = exact_solution(x, y, 0.0, alpha);
        }
    }
    u_new = u; // so u_new's boundary also starts correctly at 0

    // ---------------- Time-stepping loop ----------------
    // This is the heart of the explicit method: each new interior value depends
    // ONLY on OLD neighbouring values -- nothing here depends on values we are
    // currently computing. That's what makes this scheme trivial to parallelise later.
    for (int step = 0; step < num_steps; ++step) {
        for (int i = 1; i <= N; ++i) {
            for (int j = 1; j <= N; ++j) {
                double laplacian = (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1]
                                     - 4.0 * u[i][j]) / (h * h);
                u_new[i][j] = u[i][j] + alpha * dt * laplacian;
            }
        }
        // The boundary values of u_new are never touched by the loop above, and they
        // were set to 0 once at the start -- so they correctly stay 0 forever.
        std::swap(u, u_new); // u now holds this step's freshly computed solution
    }

    // ---------------- Verification against the exact solution ----------------
    double actual_t_final = num_steps * dt;
    double max_error = 0.0;
    double l2_sum = 0.0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double x = i * h;
            double y = j * h;
            double exact = exact_solution(x, y, actual_t_final, alpha);
            double error = std::abs(u[i][j] - exact);
            max_error = std::max(max_error, error);
            l2_sum += error * error;
        }
    }
    double l2_error = std::sqrt(l2_sum / (size * size));

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Simulated up to t    = " << actual_t_final << "\n";
    std::cout << "Max error vs exact   = " << max_error << "\n";
    std::cout << "L2 error vs exact    = " << l2_error << "\n";

    return 0;
}