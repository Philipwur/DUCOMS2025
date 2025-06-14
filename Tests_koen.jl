# Julia script to solve the 1D shallow water equations as a DAE problem
using NonlinearSolve, LinearAlgebra, Parameters, Plots, Sundials

# This code is a template for solving the 1D shallow water equations (SWE) as a DAE problem.
# The setup is as follows:
# 1. Define parameters for the simulation, including gravity, number of grid points,
#   spatial domain, and bottom topography.
# 2. Set up initial conditions for water height and momentum.
# 3. Define the DAE residual function that describes the SWE.
# 4. Implement a time loop to solve the DAE problem using Sundials' IDA solver.
# 5. Plot the results.
# 6. Function calls to start the simulation.


@with_kw struct SWEParameters
    g::Float64                      # gravity
    cf::Float64                     # friction coefficient
    N::Int                          # number of grid points
    xspan::Tuple{Float64, Float64}  # spatial domain
    tstart::Float64                 # simulation start time
    tstop::Float64                  # simulation end time
    D::Float64                      # reference depth
    use_wavy_bed::Bool              # choose between flat or wavy bed
    bc_type::Symbol                 # Options: :periodic, :dirichlet_neumann, :gabc
    x::Vector{Float64}              # spatial grid
    zb::Vector{Float64}             # bed
end

# --- 1. Parameter setup ---

function generate_grid(N, xspan)
    x_min, x_max = xspan
    x = range(x_min, x_max; length=N)
    return x
end

function bed_profile(x, use_wavy_bed, D)
    N = length(x)
    zb = -D .* ones(N)
    if use_wavy_bed
        zb .+= 0.4 .* sin.(2π .* x .* (N - 1) ./ (N * (x[end] - x[1]) * 5))
    end
    return zb
end

function make_parameters(; N=200, xspan=(0.0, 5.0), D=10.0, g=9.81, cf=0.003,
                          tstart=0.0, tstop=1.0, use_wavy_bed=true, bc_type=:periodic)
    x = generate_grid(N,xspan)
    zb = bed_profile(x, use_wavy_bed, D)

    return SWEParameters(g, cf, N, xspan, tstart, tstop, D, use_wavy_bed, bc_type, x, zb)
end

# --- 2. Initial condition ---
function initial_conditions(params::SWEParameters)
    xmin, xmax = params.xspan

    ζ = 0.1 .* exp.(-100 .* ((params.x ./ xmax .- 0.5) .* xmax).^2)
    h = ζ .- params.zb  # total water depth
    q = zeros(length(params.x))  # initial discharge

    return h, q
end

# --- 3. DAE residual function ---
# Note: the "!" at the end of the function name indicates that the function modifies 
# its arguments (convention in Julia)
function swe_dae_residual!(residual, du, u, p::SWEParameters, t)
    N = p.N
    h = @view u[1:N]
    q = @view u[N+1:2N]
    dhdt = @view du[1:N]
    dqdt = @view du[N+1:2N]

    x = p.x
    zb = p.zb
    Δx = x[2] - x[1]
    ζ = h .+ zb

    # calculate residual, mind boundary conditions
    # ...



    residual[...] = # set up residual here for all points or control volumes etc.
    return nothing
end

# --- 4. Time integration ---
function timeloop(params)
    # Unpack parameters 

    # set up initial conditions
    h0, q0 = initial_conditions(params)
    u0 = vcat(h0, q0)
    du0 = zeros(2N)  # Initial guess for du/dt

    tspan = (tstart, tstop) # defines the start and end times for the simulation

    # Specify differentiable variables as (true) -> all variables
    differential_vars = trues(2N)

    dae_prob = DAEProblem(
        swe_dae_residual!, du0, u0, tspan, params;
        differential_vars=differential_vars
    )
    sol = solve(dae_prob, IDA(), reltol=1e-8, abstol=1e-8) # solves the DAE problem using default settings

    # --- 5. a Live Plots ---

    return sol # return solution object
end


# --- 5. b Plotting results ---


# --- 6. Main script ---
# Set up parameters
params = make_parameters()
# # Call the time loop function
solution = timeloop(params)

println(params)

h,q = initial_conditions(params)
size(h)
size(q)