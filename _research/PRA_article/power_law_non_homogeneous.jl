using DrWatson, LinearAlgebra, Optim, JLD, BSON, LineSearches, DataFrames
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
JPL(αα) = [(i!=j)*abs(i-j)^(-Float64(αα)) for i=1:Nions, j=1:Nions]
Jexp(parms) = Jexp1D(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+1], planes=[1], hessian=hess)[1];
λp(parms) = Jexp1D(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+1], planes=[1], hessian=hess)[2];
Jsoln(solution) = Jexp(solution)./maximum(abs.(Jexp(solution)))


"========================="
# Objective function and constrains
"========================="


# Seed and constrains
Ωpin(parms) = vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2]))
initial_μ = 8.0;
initial_parms = vcat(0.05*ones(Nions÷2), initial_μ);
Ωmax = 5.0; μmin = 0.5; μmax = 20.0;
lc = append!(zeros(Nions÷2), μmin);
uc = append!(Ωmax*ones(Nions÷2), μmax);

# Objective functions

DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./(abs(coupling_matrix[1,2]))
DividebyDiag(coupling_matrix::Array{Float64,2}) = coupling_matrix./(sum(diag(coupling_matrix,1))/11)

ϵ_J1(parms, α) = norm(DividebyMax(Jexp(parms)) - JPL(α))
ϵ_J2(parms, α) = 10*norm(normalize(Jexp(parms)) - normalize(JPL(α)))
ϵ_J3(parms, α) = norm(DividebyDiag(Jexp(parms)) - JPL(α))
ϵs_cons = (ϵ_J1, ϵ_J2, ϵ_J3)

ϵ_Frobenius(parms) = norm(JPL(α)-parms[Nions÷2+2]*Jexp(parms),2) #Frobenius norm
ϵ_Nuclear(parms) = nucnorm(JPL(α)-parms[Nions÷2+2]*Jexp(parms)) #Nuclear norm
ϵ_Spectral(parms) = specnorm(JPL(α)-parms[Nions÷2+2]*Jexp(parms)) #Nuclear norm
ϵs_norm = (ϵ_Frobenius, ϵ_Nuclear, ϵ_Spectral)


"========================="
# Optimization and Benchmarking
"========================="

# Algorithms
line_search = (HagerZhang(), BackTracking(), BackTracking(order=2));
algorithm(n) = (LBFGS(linesearch = line_search[n]), BFGS(linesearch = line_search[n]), GradientDescent(linesearch = line_search[n]), ConjugateGradient(linesearch = line_search[n]))

a=1;
powerlaw_LBFGS_summmary = [];
powerlaw_LBFGS_solution = [];
for α=1:0.5:4
    for e in 1:length(ϵs_cons), l in 2:length(line_search)
        ϵobj(parms) = ϵs_cons[e](parms, α)
        try 
            solution = Optim.optimize(ϵobj, lc, uc, initial_parms, Fminbox(algorithm(l)[a]), Optim.Options(time_limit = 500.0, store_trace=true))
        catch error
            println(error)
        end
        summary_sol = Dict(:target => "α="*string(α), :method => summary(solution), :objective => labels[e], :linesearch => line_search[l], :minimizer => Array(solution.minimizer), :minimum => solution.minimum, :iterations => solution.iterations, :time => solution.time_run)
        push!(powerlaw_LBFGS_summmary, summary_sol)
        push!(powerlaw_LBFGS_solution, solution)
    end
end

param_optimizer = [];
for e in 1:length(ϵs_cons), l in 2:length(line_search)
    push!(param_optimizer,[e,l])
end


"========================="
# Plot results
"========================="

n=1
for n in 1:6
    λexp=sqrt.(λp(powerlaw_LBFGS_solution[n].minimizer));
    note=powerlaw_LBFGS_summmary[n][:target]*"\n"*powerlaw_LBFGS_summmary[n][:objective]*"_"*powerlaw_LBFGS_summmary[n][:method]*"_"*string(powerlaw_LBFGS_summmary[n][:linesearch])[1:1]*"_1.0x\n[Ωᵢ,μ] = "*string(round.(powerlaw_LBFGS_summmary[n][:minimizer],digits=2))*"\nωₚ = "*string(round.(λexp, digits=2))*"\n time(s):"*string(round(powerlaw_LBFGS_summmary[n][:time],digits=1))
    PlotMatrixArticle(Jsoln(powerlaw_LBFGS_solution[n].minimizer), comment="note");display(gcf())
end

using DataFrames
power_law_NH= collect_results(datadir("PowerLaw_NH"))


DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./(maximum(coupling_matrix))

function BestMatch(result_matrix)
    αs = 1:0.25:4
    errors=[norm(DividebyMax(result_matrix)-JPL(α))/norm(JPL(α)) for α in αs]
    αindex = indexin(minimum(errors),errors)
    return minimum(errors), αs[αindex][1]
end

names(power_law_NH)
[powerlaw_LBFGS_summmary[6*n][:target] for n=1:7]

power_law_NH[!,["objective","target"]]
power_law_NH[1:21,["objective","target"]]
power_law_NH[22:42,["objective","target"]]

Jsoln(power_law_NH[36,"minimizer"])-power_law_NH[5,"Jexp"]
power_law_NH[22:42,["target","minimum"]]


[power_law_NH[n+1,"minimizer"]-power_law_NH[n,"minimizer"] for n=1:6]



Je(n)=Jsoln(powerlaw_LBFGS_solution[6*n-4].minimizer)
errors_fun(n,α) = norm(DividebyMax(Je(n))-JPL(α))/norm(JPL(α))
errs_fit=[errors_fun(m,αs[m]) for m=1:7]
error_PLNH=Array{Any}(nothing,8,2)
error_PLNH[1,:]= vcat("#α",labels[3])
error_PLNH[2:end,:]=hcat(αs, errs_fit)



Je1(n)=Jsoln(powerlaw_LBFGS_solution[n].minimizer)
Je2(n)=Jsoln(powerlaw_LBFGS_solution[n+7].minimizer)
Je3(n)=Jsoln(powerlaw_LBFGS_solution[n+14].minimizer)

Je(n)=Jsoln(powerlaw_LBFGS_solution[6*n-4].minimizer)

errs_fit=[errors_fun(m,αs[m]) for m=1:7]


err_fun1, α_fun1 = [BestMatch(Je3(n)) for n=1:7]

errors_fun1(n,α) = norm(DividebyMax(Je1(n))-JPL(α))/norm(JPL(α))
errors_fun2(n,α) = norm(DividebyMax(Je2(n))-JPL(α))/norm(JPL(α))
errors_fun3(n,α) = norm(DividebyMax(Je3(n))-JPL(α))/norm(JPL(α))
errors_fun(n,α) = norm(DividebyMax(Je(n))-JPL(α))/norm(JPL(α))

errs_fit1=[errors_fun1(m,αs[m]) for m=1:7]
errs_fit2=[errors_fun2(m,αs[m]) for m=1:7]
errs_fit3=[errors_fun3(m,αs[m]) for m=1:7]

error_PLNH=Array{Any}(nothing,8,2)
error_PLNH[1,:]= vcat("#α",labels[3])
error_PLNH[2:end,:]=hcat(αs, errs_fit)

writedlm(datadir("PowerLaw_NH")*"/error_power_law_non_homogenous.txt", error_PLNH)

# Solutions for norm(Jₜ/max(Jₜ)-Jₑ/max(Jₑ))
power_law[4:16,[:errorfunction,:model]]
JPLexpE1(n) = power_law[4:16, :Jexp][n]


err_fun1, α_fun1 = [BestMatch(JPLexpE1(n)) for n=1:13]

errors_fun1(n,α) = norm(DividebyMax(JPLexpE1(n))-JPL(α))/norm(JPL(α))

αs = collect(1:0.5:4);

errs_fit1=[errors_fun1(m,αs[m]) for m=1:13]