using DrWatson, LinearAlgebra, Optim, JLD, BSON, LineSearches, PyPlot
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using optimizers, coupling_matrix, crystal_geometry #plotting_functions



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
#* Frequencies
ωtrap = 2π*MHz*[0.6, 0.6, 0.1];

"========================="
# Positions and Hessian
"========================="

pos_ions = PositionIons(Nions, ωtrap, plot_position=false, tcool=500E-6, cvel=10E-20)
pos_ions = Chop_Sort(pos_ions)


function PositionEvolution(ref_positions, χ)
    new_positions = zero(ref_positions)
    d₀ = ref_positions[3,Nions÷2] - ref_positions[3,Nions÷2 + 1]
    dₙ = [ref_positions[3,i]-ref_positions[3,i+1] for i=Nions÷2:11]
    Δₙ = dₙ .- d₀
    dₙ̃ = (1-χ)*Δₙ .+ d₀
    xₙ = [ref_positions[3,Nions÷2 + 1] + sum([dₙ̃[n] for n=1:m]) for m=1:Nions÷2]
    new_positions[3,:] = prepend!(-xₙ, reverse(xₙ))
    return new_positions
end

Hessian_X(pos) = Hessian(pos, ωtrap; planes=[1])
χs = collect(0:0.1:1)
fig, ax = plt.subplots()
for n=1:11
    χ = χs[n]
    scatter(PositionEvolution(pos_ions, χ)[3,:], n*ones(12),c="black",label=string(χ))
end
legend()
gcf()

"========================="
# Optimization and Benchmarking
"========================="

function PowerLaw_LBGFS(α, hessian; order_BT=3)
    #* Seed and constrains
    Ωpin(parms) = vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2]))
    μ_seed = 2.0;
    parms_seed = vcat(0.05*ones(Nions÷2), μ_seed);
    Ωmax = 20.0; 
    μmin = 0.5; 
    μmax = 20;
    lc = append!(zeros(Nions÷2), μmin);
    uc = append!(Ωmax*ones(Nions÷2), μmax);
    
    #* Target and experimental matrix
    JPL = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]
    Jexp(parms) = Jexp1D(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+1], planes=[1], hessian=hessian)[1];

    #* Objective function
    DividebyDiag(coupling_matrix::Array{Float64,2}) = coupling_matrix./(sum(diag(coupling_matrix,1))/11)
    ϵ_J(parms) = norm(DividebyDiag(Jexp(parms)) - JPL)

    #* Solve
    solution = Optim.optimize(ϵ_J, lc, uc, parms_seed, Fminbox(LBFGS(linesearch = BackTracking(order=order_BT))), Optim.Options(time_limit = 500.0, store_trace=true))

    return solution, Jexp(solution.minimizer)
end

SummarySolution(solution, coupling_matrix, target, objective, linesearch) = Dict(:target => target, :method => summary(solution), :objective => objective, :linesearch => linesearch, :minimizer => Array(solution.minimizer), :minimum => solution.minimum, :iterations => solution.iterations, :time => solution.time_run, :ωtrap => Array(ωtrap), :Jexp => Array(coupling_matrix), :axis => "x")
    


#* Optimization
collection_matrices = [];
collection_summary = [];
collection_solutions = [];

χs = collect(0:0.1:1)
for αₜ=0.5:0.5:4
    coupling_matrices = zeros(12,12,11);
    powerlaw_LBFGS_summmary = [];
    powerlaw_LBFGS_solution = [];
    for n=1:11
        χ = χs[n]
        hessχ = Hessian_X(PositionEvolution(pos_ions, χ));
        solution, Jexp = PowerLaw_LBGFS(αₜ, hessχ, order_BT=3)
        summary_sol = SummarySolution(solution, Jexp, "PL_α="*string(αₜ), "||Jₜ-Jₑ÷ave(diag(Jₑ,1))||","BT_order=3")
        push!(powerlaw_LBFGS_summmary, summary_sol)
        push!(powerlaw_LBFGS_solution, solution)
        coupling_matrices[:,:,n] = Jexp
    end
    push!(collection_matrices, coupling_matrices)
    push!(collection_solutions, powerlaw_LBFGS_solution)
    push!(collection_summary, powerlaw_LBFGS_summmary)
end


"==============="
# Error analysis
"==============="

Jsoln(solution) = Jexp(solution)./maximum(abs.(Jexp(solution)))
JPL(α) = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]

ϵ_J(matrix, α) = opnorm(matrix./maximum(abs.(matrix)) - JPL(α))/opnorm(JPL(α))

αs = collect(0.5:0.5:4)
ϵ_vs_α_vs_χ = zeros(8,11)
for k=1:8
    ϵ_vs_α_vs_χ[k,:] = [ϵ_J(collection_matrices[k][:,:,n], αs[k]) for n=1:11]
end


fig = plt.figure()
gs = fig.add_gridspec(4,2, hspace=0.07)
axs = gs.subplots(sharex=true)
k=0
for n=1:4, m=1:2
    k+=1
    min_lim = 0.95*minimum(ϵ_vs_α_vs_χ[k,:][10:11])
    max_lim = 1.05*maximum(ϵ_vs_α_vs_χ[k,:][1:2])
    axs[n,m].scatter(0:0.1:1,ϵ_vs_α_vs_χ[k,:], label="α="*string(αs[k]), s=10);axs[n,m].set_ylim(min_lim, max_lim); axs[n,m].legend();
end

axs[4,1].set_xlabel("χ");
axs[4,2].set_xlabel("χ");
axs[1,1].set_ylim(0.05, 0.1);
axs[1,2].set_ylim(0.07, 0.1);
axs[2,2].set_ylim(0.0365, 0.039);
axs[3,1].set_ylim(0.01, 0.1);
axs[4,2].set_ylim(0.10, 0.11);

savefig(plotsdir("HomtoInhom","LBFGS_BT_Diag_Div.png"));gcf()


"============"
# Export data
"============"

for n=1:8, m=1:11
    dict_sol = collection_summary[n]
    wsave(datadir("HomtoInhom", savename(dict_sol, "bson")), dict_sol)
end


size(collection_summary[1])
collection_summary[1][88]



