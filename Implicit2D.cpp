// Implicit2D.cpp
//
// Sequential 2D heat equation solver using the Implicit (Backward Euler) scheme,
// with the resulting linear system solved by Jacobi iteration at every time step.
//
// Solves:   du/dt = alpha * (d2u/dx2 + d2u/dy2)   on the unit square [0,1] x [0,1]
// with homogeneous Dirichlet boundary conditions (u = 0 on all four edges).
//
// Verified against the SAME exact manufactured solution used for the explicit version,
// so the two files can be compared directly:
//
//     u_exact(x, y, t) = sin(pi*x) * sin(pi*y) * exp(-2 * alpha * pi^2 * t)
//
// The key difference from the explicit version: here, the new value at every point
// depends on the NEW values of its neighbours too (not just the old ones). That means
// we cannot just plug numbers into one formula -- we have to solve a whole system of
// equations at every time step. Jacobi iteration does this by guessing, checking,
// and improving the guess, repeatedly, until it stops changing -- the same idea as
// the "two rooms heating each other" example: guess a value, use it to update the
// neighbours, then use the neighbours' updated values to correct the guess, and repeat.

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

const double PI = 3.14159265358979323846;

double exact_solution(double x, double y, double t, double alpha) {
    return std::sin(PI * x) * std::sin(PI * y) * std::exp(-2.0 * alpha * PI * PI * t);
}

int main() {
    // ---------------- Problem parameters ----------------
    const double alpha    = 1.0;
    const int    N        = 40;
    const double L        = 1.0;
    const double h        = L / (N + 1);
    const double t_final  = 0.05;

    // Implicit scheme has NO stability limit on r -- that is the whole point of it.
    // We can pick a much bigger time step than the explicit scheme was allowed to use.
    const double r         = 1.0;   // deliberately well above the explicit limit of 0.25
    const double dt         = r * h * h / alpha;
    const int    num_steps  = static_cast<int>(t_final / dt);

    // ---------------- Jacobi iteration settings ----------------
    const int    max_jacobi_iters = 10000; // safety cap, should converge well before this
    const double jacobi_tolerance = 1e-10; // stop once the guess barely changes anymore

    std::cout << "=== 2D Implicit Heat Equation Solver (Backward Euler + Jacobi) ===\n";
    std::cout << "Grid spacing h       = " << h << "\n";
    std::cout << "Time step dt         = " << dt << "\n";
    std::cout << "Stability number r   = " << r << "  (no upper limit needed here)\n";
    std::cout << "Number of time steps = " << num_steps << "\n\n";

    // ---------------- Storage ----------------
    int size = N + 2;
    std::vector<std::vector<double>> u(size, std::vector<double>(size, 0.0));      // current time level (known)
    std::vector<std::vector<double>> u_old(size, std::vector<double>(size, 0.0));  // same as u, kept fixed during Jacobi
    std::vector<std::vector<double>> u_new(size, std::vector<double>(size, 0.0));  // next Jacobi guess

    // ---------------- Initial condition ----------------
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double x = i * h;
            double y = j * h;
            u[i][j] = exact_solution(x, y, 0.0, alpha);
        }
    }

    // ---------------- Time-stepping loop ----------------
    for (int step = 0; step < num_steps; ++step) {
        // u_old holds this step's STARTING values -- these stay fixed while Jacobi runs.
        u_old = u;
        // Start the Jacobi guess from the old values -- a reasonable first guess.
        u_new = u;

        // ---- Jacobi iteration: guess, check, improve, repeat ----
        for (int iter = 0; iter < max_jacobi_iters; ++iter) {
            double max_change = 0.0;
            std::vector<std::vector<double>> u_guess = u_new; // last round's guess, used to compute this round

            for (int i = 1; i <= N; ++i) {
                for (int j = 1; j <= N; ++j) {
                    // Backward Euler, rearranged to solve for u_new(i,j):
                    //   u_new(i,j) = [ u_old(i,j) + r*(neighbours of u_new, from LAST guess) ] / (1 + 4r)
                    double neighbour_sum = u_guess[i+1][j] + u_guess[i-1][j]
                                          + u_guess[i][j+1] + u_guess[i][j-1];
                    double updated = (u_old[i][j] + r * neighbour_sum) / (1.0 + 4.0 * r);
                    max_change = std::max(max_change, std::abs(updated - u_guess[i][j]));
                    u_new[i][j] = updated;
                }
            }
            // Boundaries stay at 0 throughout -- never touched, matches Dirichlet BC.

            if (max_change < jacobi_tolerance) {
                break; // the guess has settled -- no need to keep iterating
            }
        }

        u = u_new; // move on to the next time step
    }

    // ---------------- Verification against the exact solution ----------------
    double actual_t_final = num_steps * dt;
    double max_error = 0.0, l2_sum = 0.0;
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