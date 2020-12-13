using DrWatson, LinearAlgebra, Optim, JLD, BSON, LineSearches
@quickactivate "IonTweezers"
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
using coupling_matrix, crystal_geometry

"================"
# Definitions
"================"
MHz = 1E6; μm = 1E-6;
Chop_Sort(x) = sort(round.(x,digits=5), dims=2)
DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./maximum(coupling_matrix)

"========================="
# Crystal parameters
"========================="

Nions = 12;
### Frequencies ###
ωtrap = 2π*MHz*[0.6, 4.0, 0.4];

"========================="
# Positions and Hessian
"========================="

pos_ions = PositionIons(Nions,ωtrap,plot_position=false)
pos_ions=Chop_Sort(pos_ions)

hess=Hessian(pos_ions, ωtrap;planes=[1])


"========================="
# Loop over different power-law decays
"========================="

# Target and experimental matrices and phonon frequencies
JPL(α) = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]
JPLexp(parms) = Jexp1D(pos_ions, vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2])), ωtrap, parms[Nions÷2+1], planes=[1], hessian=hess)[1:2];

# Seed and constrains
initial_μ = 2.0;
initial_parms = vcat(0.05*ones(Nions÷2), initial_μ);
Ωmax = 1.0; μmin = 0.5; μmax = 20;
lc = append!(zeros(Nions÷2), μmin);
uc = append!(Ωmax*ones(Nions÷2), μmax);

# Power law coefficients
αs = collect(1:0.5:4);
solution_1 = zeros(7,13);
solution_2 = zeros(7,13);

for n in 1:7
    α=αs[n]

    # Objective functions
    ϵ_J1(parms) = norm(DividebyMax(JPLexp(parms)[1]) - JPL(α))
    println(n)
    
    #Optimization
    #inner_optimizer= LBFGS(linesearch = BackTracking(order=2))
    inner_optimizer= LBFGS()
    solutionGD_1 = Optim.optimize(ϵ_J1, lc, uc, initial_parms, Fminbox(inner_optimizer))
	println("done with obj function 1")
    solution_1[:,n]= solutionGD_1.minimizer
    
    #Collect parameters
    parms_1 = Dict(:ωtrap => ωtrap, :Ωpin => solution_1[:,n][1:Nions÷2], :μnote => solution_1[:,n][Nions÷2 + 1], :axis => "x", :homogeneous => false, :model => replace("r^(-\$α)", "\$α" => "$α"), :optimizer => Dict(:method => "LBFGS", :objective => "norm(Jₜ/max(Jₜ)-Jₑ/max(Jₑ))"), :errorfunction => "1")
    parms_1[:Jexp] = JPLexp(solution_1[:,n])[1]; parms_1[:ωm] = JPLexp(solution_1[:,n])[2]
    
    #Save solution and parameters
    wsave(datadir("PowerLaw", savename(parms_1, "bson")), parms_1)
end


