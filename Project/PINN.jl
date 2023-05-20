using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim

@variables x[1:2] # Define x as a 2-dimensional interval
@variables u(x[1], x[2])

@derivatives Dxx'' ~ x[1]
@derivatives Dyy'' ~ x[2]

# 2D PDE
eq = Dxx(u(x[1], x[2])) + Dyy(u(x[1], x[2])) ~ -sin(pi * x[1]) * sin(pi * x[2])

# Boundary conditions
bcs = [u(0, x[2]) ~ 0.0, u(1, x[2]) ~ 0.0,
    u(x[1], 0) ~ 0.0, u(x[1], 1) ~ 0.0]

# Neural network
dim = 2 # number of dimensions
chain = FastChain(FastDense(dim, 16, Flux.σ), FastDense(16, 16, Flux.σ), FastDense(16, 1))

# Discretization
dx = 0.05
discretization = PhysicsInformedNN(chain, GridTraining(dx))

@named pde_system = PDESystem(eq, bcs, [x[1], x[2]], [u(x[1], x[2])])
prob = discretize(pde_system, discretization)

# Optimizer
opt = Optim.BFGS()

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, opt, callback = callback, maxiters = 1000)
phi = discretization.phi

xs, ys = [infimum(d.domain):(dx / 10):supremum(d.domain) for d in prob.domain]
analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

u_predict = reshape([first(phi([x, y], res.u)) for x in xs, y in ys], (length(xs), length(ys)))
u_real = reshape([analytic_sol_func(x, y) for x in xs, y in ys], (length(xs), length(ys)))
diff_u = abs.(u_predict .- u_real)

using Plots

p1 = plot(xs, ys, u_real, linetype = :contourf, title = "analytic")
p2 = plot(xs, ys, u_predict, linetype = :contourf, title = "predict")
p3 = plot(xs, ys, diff_u, linetype = :contourf, title = "error")
plot(p1, p2, p3)
