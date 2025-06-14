# CScNL hackathon 2025, team QAIMS

Direct implementation of solver: `src/main_without_smoothing.jl`

Implementation with added MUSCL (Monotonic upstream-centered schemes for conservation laws) and Rusanov-flux: `src/main_with_smoothing.jl`

Example animations in folder `results/`
- periodic boundary conditions without and with smoothing
- Dirichlet-Neuman boundary conditions with $q_{in} = 1.0$ and $\zeta_{out} = -1.0$ without and with smooting

