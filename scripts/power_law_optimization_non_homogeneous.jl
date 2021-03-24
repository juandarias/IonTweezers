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
ωtrap = 2π*MHz*[0.6, 0.6, 0.1];

"========================="
# Positions and Hessian
"========================="

pos_ions = PositionIons(Nions,ωtrap,plot_position=false, tcool=500E-6, cvel=10E-20)
pos_ions=Chop_Sort(pos_ions)

hess=Hessian(pos_ions, ωtrap; planes=[1])


"========================="
# Target and experimental matrix
"========================="

# Target and experimental matrices and phonon frequencies
JPL(α) = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]
Jexp(parms) = Jexp1Dc(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+1], planes=[1], hessian=hess)[1];
λp(parms) = Jexp1Dc(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+1], planes=[1], hessian=hess)[2];
Jsoln(solution) = Jexp(solution)./maximum(abs.(Jexp(solution)))


"========================="
# Objective function and constrains
"========================="


# Seed and constrains
Ωpin(parms) = vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2]))
initial_μ = 2.0;
initial_parms = vcat(0.05*ones(Nions÷2), initial_μ);
Ωmax = 5.0; μmin = 0.5; μmax = 20;
lc = append!(zeros(Nions÷2), μmin);
uc = append!(Ωmax*ones(Nions÷2), μmax);

# Objective functions

DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./(abs(coupling_matrix[1,2]))
DividebyDiag(coupling_matrix::Array{Float64,2}) = coupling_matrix./(sum(diag(coupling_matrix,1))/11)

ϵ_J1(parms) = norm(DividebyMax(Jexp(parms)) - JPL(α))
ϵ_J2(parms) = 10*norm(normalize(Jexp(parms)) - normalize(JPL(α)))
ϵ_J3(parms) = norm(DividebyDiag(Jexp(parms)) - JPL(α))
ϵs_cons = (ϵ_J1, ϵ_J2, ϵ_J3)

ϵ_Frobenius(parms) = norm(JPL(α)-parms[Nions÷2+2]*Jexp(parms),2) #Frobenius norm
ϵ_Nuclear(parms) = nucnorm(JPL(α)-parms[Nions÷2+2]*Jexp(parms)) #Nuclear norm
ϵ_Spectral(parms) = specnorm(JPL(α)-parms[Nions÷2+2]*Jexp(parms)) #Nuclear norm
ϵs_norm = (ϵ_Frobenius, ϵ_Nuclear, ϵ_Spectral)


"========================="
# Optimization and Benchmarking
"========================="


a=1;
powerlaw_LBFGS_summmary = [];
powerlaw_LBFGS_solution = [];
for α=1:0.5:4
    for e in 1:length(ϵs_cons), l in 1:length(line_search)
        try 
            solution = Optim.optimize(ϵs_cons[e], lc, uc, initial_parms, Fminbox(algorithm(l)[a]), Optim.Options(time_limit = 500.0, store_trace=true))
        catch error
            println(error)
        end
        summary_sol = Dict(:target => "α="*string(α), :method => summary(solution), :objective => string(ϵs_cons[e]), :linesearch => line_search[l], :minimizer => Array(solution.minimizer), :minimum => solution.minimum, :iterations => solution.iterations, :time => solution.time_run)
        push!(powerlaw_LBFGS_summmary, summary_sol)
        push!(powerlaw_LBFGS_solution, solution)
    end
end


"========================="
# Plot results
"========================="


for n in 1:6
    λexp=sqrt.(λp(powerlaw_LBFGS_solution[n].minimizer));
    note=powerlaw_LBFGS_summmary[n][:target]*"\n"*powerlaw_LBFGS_summmary[n][:objective]*"_"*powerlaw_LBFGS_summmary[n][:method]*"_"*string(powerlaw_LBFGS_summmary[n][:linesearch])[1:1]*"_1.0x\n[Ωᵢ,μ] = "*string(round.(powerlaw_LBFGS_summmary[n][:minimizer],digits=2))*"\nωₚ = "*string(round.(λexp, digits=2))*"\n time(s):"*string(round(powerlaw_LBFGS_summmary[n][:time],digits=1))
    PlotMatrixArticle(Jsoln(powerlaw_LBFGS_solution[n].minimizer), comment=note);display(gcf())
end





