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
    prescribed_inlet_discharge::Union{Nothing, Function} # for dirichlet_neumann
    prescribed_outlet_surface::Union{Nothing, Function} # for dirichlet_neumann
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
                          tstart=0.0, tstop=1.0, use_wavy_bed=true, bc_type=:periodic,
                          prescribed_inlet_discharge=nothing, prescribed_outlet_surface=nothing)
    x = generate_grid(N,xspan)
    zb = bed_profile(x, use_wavy_bed, D)

    if bc_type == :dirichlet_neumann
        if prescribed_inlet_discharge === nothing || prescribed_outlet_surface === nothing
            error("Both `prescribed_inlet_discharge` and `prescribed_outlet_surface` must be provided for `:dirichlet_neumann` BC.")
        end
    end

    return SWEParameters(g, cf, N, xspan, tstart, tstop, D, use_wavy_bed, bc_type, x, zb,
        prescribed_inlet_discharge, prescribed_outlet_surface
    )
end

# --- 2. Initial condition ---
function initial_conditions(params::SWEParameters)
    xmin, xmax = params.xspan

    ζ = 0.1 .* exp.(-100 .* ((params.x ./ xmax .- 0.5) .* xmax).^2)
    h = ζ .- params.zb  # total water depth
    q = zeros(length(params.x))  # initial discharge

    return h, q
end

function plot_solution(h, q, params::SWEParameters, filename=nothing)
    if isnothing(filename)
        filename = "plot.png"
    end

    N = params.N
    x = params.x
    zb = params.zb

    ζ = h .+ zb

    # Top plot: zeta and zb
    # p1 = plot(x, ζ; label="ζ(x)", xlabel="x", ylabel="Surface Level", lw=2, legend=:topright)
    # plot!(p1, x, zb; label="zb(x)", lw=1, color=:gray)

    p1 = plot(x, ζ; label="ζ(x)", xlabel="x", ylabel="Surface Level", color=:blue, fillrange=zb, 
              fillalpha=0.3,lw=2, legend=:topright)
    bed_base = minimum(zb) - 0.5
    plot!(p1, x, zb; label="zb(x)", lw=1, color=:brown, fillrange=bed_base, fillalpha=0.4)

    # Bottom plot: discharge
    p2 = plot(x, q; label="q(x)", xlabel="x", ylabel="Discharge", lw=2, legend=:topright)

    final_plot = plot(p1, p2; layout=(2,1), size=(800,600))

    savefig(final_plot, filename)

    return final_plot
    
end


function animate_solution(h_all, q_all, params::SWEParameters; 
                          filename::String="swe_animation.gif", 
                          framerate::Int=20,
                          skip::Int=1)
    @assert size(h_all) == size(q_all) "h and q must have the same shape"

    nt, N = size(h_all)
    @assert N == params.N "Mismatch in spatial dimension"

    anim = @animate for i in 1:skip:nt
        h = view(h_all, i, :)
        q = view(q_all, i, :)
        framefile = tempname() * ".png"
        plot_solution(h, q, params, framefile)
    end

    gif(anim, filename, fps=framerate)
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

    # For periodic boundary conditions:
    if p.bc_type == :periodic
        # Apply periodic ghost cells
        h_ext = [h[end]; h; h[1]]
        q_ext = [q[end]; q; q[1]]
        zb_ext = [zb[end]; zb; zb[1]]
        ζ_ext = h_ext .+ zb_ext

        
        # Spatial derivatives (central difference)
        dqdx = @. (q_ext[3:end] - q_ext[1:end-2]) / (2Δx)
        residual[1:N] .= dhdt .+ dqdx

        # is this second order accurate?
        q2_over_h = @. q_ext[2:end-1]^2 / h_ext[2:end-1]
        q2_over_h_ext = [q2_over_h[end]; q2_over_h; q2_over_h[1]]
        dfluxdx = @. (q2_over_h_ext[3:end] - q2_over_h_ext[1:end-2]) / (2Δx)

        dzetadx = @. (ζ_ext[3:end] - ζ_ext[1:end-2]) / (2Δx)
        friction = @. p.cf * q .* abs(q) / h.^2

        residual[N+1:2N] .= dqdt .+ dfluxdx .+ p.g .* h .* dzetadx .+ friction
    elseif p.bc_type == :dirichlet_neumann
        # Prescribed BCs
        qL = p.prescribed_inlet_discharge(t)
        ζR = p.prescribed_outlet_surface(t)

        # Extend fields with ghost cells
        q_ext = [qL; q; q[end]]                        # Dirichlet on q at left
        ζ_ext = [ζ[1]; ζ; ζR]                          # Dirichlet on ζ at right
        h_ext = [h[1]; h; ζR - zb[end]]                # derive h ghost from ζR - zb
        zb_ext = [zb[1]; zb; zb[end]]                  # Neumann (copy value)

        # ∂q/∂x (mass conservation)
        dqdx = @. (q_ext[3:end] - q_ext[1:end-2]) / (2Δx)
        residual[1:N] .= dhdt .+ dqdx

        # convective term ∂/∂x (q²/h)
        q2_over_h = @. q_ext[2:end-1]^2 / h_ext[2:end-1]
        q2_over_h_ext = [q2_over_h[1]; q2_over_h; q2_over_h[end]]  # Neumann-like
        dfluxdx = @. (q2_over_h_ext[3:end] - q2_over_h_ext[1:end-2]) / (2Δx)

        # surface gradient ∂ζ/∂x
        dzetadx = @. (ζ_ext[3:end] - ζ_ext[1:end-2]) / (2Δx)

        # friction
        friction = @. p.cf * q * abs(q) / h^2

        residual[N+1:2N] .= dqdt .+ dfluxdx .+ p.g .* h .* dzetadx .+ friction

    end
    
    return nothing
end


# --- 4. Time integration ---
function timeloop(params)
    # Unpack parameters 
    N = params.N
    tstart = params.tstart
    tstop = params.tstop

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
qin(t) = 1.0
zout(t) = -1.0

params = make_parameters(;use_wavy_bed=true, bc_type=:dirichlet_neumann,
    prescribed_inlet_discharge=qin,prescribed_outlet_surface=zout
)
# # Call the time loop function
# solution = timeloop(params)

println(params)

h,q = initial_conditions(params)
size(h)
size(q)

plot_solution(h, q, params)

#%%

sol = timeloop(params)

size(sol)

h_all = transpose(sol[1:params.N,:])
q_all = transpose(sol[(params.N+1):end,:])
size(h_all)
size(q_all)

animate_solution(h_all,q_all,params,skip=50)

#%%