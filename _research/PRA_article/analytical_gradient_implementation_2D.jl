using DrWatson, PyPlot, LinearAlgebra, Optim, LineSearches, SchattenNorms
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using optimizers, coupling_matrix, crystal_geometry, plotting_functions



"========================="
# Auxiliary functions
"========================="

Chop_Sort(x) = sort(round.(x,digits=5), dims=2)
Chop_Sort2D(A) = A[:, sortperm(A[3,:])]
diagless(A) = A - diagm(diag(A))
Σ = sum
⊗(A,B) = kron(A',B)


"===================================="
#  Parameters and constants
"===================================="

MHz = 1E6; μm = 1E-6;
Nions = 12;

### Frequencies ###
ωtrap = 2π*MHz*[0.8, 0.5, 0.15];
ωtrap = 2π*MHz*[0.6, 0.4, 0.14];


"========================="
# Positions and Hessian
"========================="

pos_ions = PositionIons(Nions, ωtrap, plot_position=false, tcool=500E-6, cvel=10E-20);
pos_ions = Chop_Sort2D(pos_ions)

PositionIons(Nions,ωtrap,plot_position=true, tcool=500E-6, cvel=10E-20)

"========================="
# Target and experimental matrices
"========================="

Jladder = [i==(j+2) ? 0.5*(j!=1)*(i!=Nions) : 0 for i=1:Nions, j=1:Nions] + [i==(j+1) ? -1*(j!=1)*(i!=Nions) : 0 for i=1:Nions, j=1:Nions] + [(j==(i+1) && i in [1,Nions-1]) ? 0.5 : 0 for i=1:Nions, j=1:Nions];
Jladder = Jladder + Jladder'

hessXYZ = Hessian(pos_ions, ωtrap; planes=[1,2,3]);
hessYZ = hessXYZ[Nions+1:3*Nions,Nions+1:3*Nions];

μsol = 6.918
Jexp(parms) = Jexp2D(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+1], hessian=hessYZ)[1];
Jexp_1(parms) = Jexp2D(pos_ions, Ωpin(parms), ωtrap, μsol, hessian=hessYZ)[1];
eigsystem(parms) = Jexp2D(pos_ions, Ωpin(parms), ωtrap, μsol, hessian=hessYZ)[2:3];


"========================="
# Objective function and constrains
"========================="

# Seed and constrains
Ωpin(parms) = vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2]), parms[1:Nions÷2], reverse(parms[1:Nions÷2]))
initial_μ = 5.0; Ωmax = 5.0; μmin = 0.5; μmax = 30; cmin = 1E-2; cmax = 1E4;

# without proportionality constants
initial_parms = vcat(0.05*ones(Nions÷2), [initial_μ]);
lc = append!(zeros(Nions÷2), [μmin]);
uc = append!(Ωmax*ones(Nions÷2), [μmax]);


# without beatnote
initial_parms = vcat(0.05*ones(Nions÷2));
lc = zeros(Nions÷2);
uc = Ωmax*ones(Nions÷2);


# Objective functions
DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./(abs(coupling_matrix[1,2]))
ϵ_J2(parms) = 10*norm(normalize(Jexp_1(parms)) - normalize(Jladder))

# Auxiliary functions
let λ = eigen(hessYZ).values, U = eigen(hessYZ).vectors;
    global ∂Jᵢⱼ∂μ(i,j,μ) = 2*μ*sum([U[i,m]*U[j,m]/((μ^2-λ[m])^2) for m=1:Nions])
    global function A̅ₖₗ(k,l,μ)
        Θ(m) = (μ^2-λ[m])^-1
        F = [((i!=j)*(λ[i]-λ[j])^-1) for i in 1:length(λ), j in 1:length(λ)]
        Λ̅  = diagm([Θ(i)^2*U[k,i]*U[l,i] for i=1:length(λ)])
        U̅ = zeros(length(λ), length(λ))
        U̅ = [(k==i || l==i) && sum([Θ(j)*U[n,j] for n=1:length(λ)]) for i=1:length(λ), j=1:length(λ)]
        A̅ = U*(Λ̅  + 0.5*F.*(U'*U̅ - U̅'*U))*U'
        return A̅
    end
end

# Analytical gradient
function g!(G, parms)
    μᵢ = parms[end]
    Ωᵢ = parms[1:end-1]
    Jₑ = Jexp(parms)
    ϵⱼ = ϵ_J2(parms)
    ΔJ = Jₑ - normalize(Jladder)
    #Do I need to calculate U and λ at each iteration of the gradient? No, see section 2.2.1 of Miles
    A̅ϵ = (ϵⱼ^-1)*sum([Jₑ[i,j]*A̅ₖₗ(i,j,μᵢ) for i=1:Nions for j=1:Nions])
    for i=1:length(Ωᵢ)
        G[i] = A̅ϵ[i,i]
    end
    ∂ϵ∂μ(μ) = (ϵⱼ)^-1*sum([ΔJ[i,j]*∂Jᵢⱼ∂μ(i,j,μ) for i=1:Nions for j=1:Nions])
    G[length(parms)] = ∂ϵ∂μ(μᵢ)
end

### Analytical gradient without beatnote
# Auxiliary functions
let λ = eigen(hessYZ).values, U = eigen(hessYZ).vectors, μ = μsol;
    global function A̅ₖₗ(k,l)
        Θ(m) = (μ^2-λ[m])^-1
        F = [((i!=j)*(λ[i]-λ[j])^-1) for i in 1:length(λ), j in 1:length(λ)]
        Λ̅  = diagm([Θ(i)^2*U[k,i]*U[l,i] for i=1:length(λ)])
        U̅ = zeros(length(λ), length(λ))
        U̅ = [(k==i || l==i) && sum([Θ(j)*U[n,j] for n=1:length(λ)]) for i=1:length(λ), j=1:length(λ)]
        A̅ = U*(Λ̅  + 0.5*F.*(U'*U̅ - U̅'*U))*U'
        return A̅
    end
end



function A̅ₖₗ(k,l,U,λ)
    Θ(m) = (μsol^2-λ[m])^-1
    F = [((i!=j)*(λ[i]-λ[j])^-1) for i in 1:length(λ), j in 1:length(λ)]
    Λ̅  = diagm([Θ(i)^2*U[k,i]*U[l,i] for i=1:length(λ)])
    U̅ = zeros(length(λ), length(λ))
    U̅ = [(k==i || l==i) && sum([Θ(j)*U[n,j] for n=1:length(λ)]) for i=1:length(λ), j=1:length(λ)]
    A̅ = U*(Λ̅  + 0.5*F.*(U'*U̅ - U̅'*U))*U'
    return A̅
end



function g_noμ!(G, parms)
    Ωᵢ = parms
    Jₑ = Jexp_1(parms)
    #λ, U = eigsystem(parms)
    ϵⱼ = ϵ_J2(parms)
    #A̅ₖₗ₂(k,l) = A̅ₖₗ(k,l,U,λ)
    ΔJ = normalize(Jₑ) - normalize(Jladder)
    #Do I need to calculate U and λ at each iteration of the gradient? No, see section 2.2.1 of Miles
    A̅ϵ = (ϵⱼ^-1)*sum([ΔJ[i,j]*A̅ₖₗ(i,j) for i=1:2*Nions for j=1:2*Nions])
    for i=1:length(Ωᵢ)
        G[i] = A̅ϵ[i,i]
    end
end




"========================="
# Optimization
"========================="

solution_grad = Optim.optimize(ϵ_J2, g_noμ!, lc, uc, initial_parms, Fminbox(LBFGS(linesearch=BackTracking())), Optim.Options(time_limit = 500.0, store_trace=true))

solution.minimizer #solution with gradient, no update of eigenvectors
solution_nograd.minimizer
solution_grad.minimizer #solution with updated gradient function
