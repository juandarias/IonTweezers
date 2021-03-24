using DrWatson, PyPlot, Distributions, Random
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using optimizers, coupling_matrix, crystal_geometry, plotting_functions



"========================="
# Parameters and constants
"========================="

MHz = 1E6; μm = 1E-6;
Nions = 12; ωtrap = 2π*MHz*[2, 0.6, 0.07]; μnote = 6.62355


Chop_Sort(x) = sort(round.(x,digits=5), dims=2)
diagless(A) = A - diagm(diag(A))

"========================="
# Positions and Hessian
"========================="

pos_ions = PositionIons(Nions,ωtrap,plot_position=false);
pos_ions=Chop_Sort(pos_ions)

PositionIons(Nions,ωtrap,plot_position=true)

hess=Hessian(pos_ions, ωtrap;planes=[2]);
λ0, b0= eigen(hess);

"============================="
# Coupling matrix homogeneous
"============================="

Hom1D = Dict(:ωtrap => ωtrap, :Ωpin=> Array(sqrt.([8.81055,18.7972,35.5,40.716,48.6586,50.0153,50.0153,48.6586,40.716,35.5,18.7972,8.81055])), :μnote => 7.01356, :axis => "y", :homogeneous => false)

Jhom1D, λm, bm = Jexp1D(pos_ions, Hom1D[:Ωpin], ωtrap, Hom1D[:μnote], hessian=hess)

Jhom1D=Jhom1D./maximum(Jhom1D)

"============================="
# Random pinning potentials
"============================="

# Random distribution with mean Ωᵢ
td(Ωᵢ) = truncated(Normal(Ωᵢ, 0.5), 0.0, Inf)

n_implementations=100;
Ωpin_random = reshape(vcat([rand(td(Hom1D[:Ωpin][i]),n_implementations) for i=1:Nions]...),n_implementations,Nions)
setJ=[]; NN_coupling=[];
for n in 1:n_implementations
    Jrandom = 1E14*Jexp1D(pos_ions, Ωpin_random[n,:], ωtrap, Hom1D[:μnote], hessian=hess)[1]
    push!(setJ, Jrandom./maximum(Jrandom))
    push!(NN_coupling, diag(Jrandom,1))
end

figure()
hist(vcat(NN_coupling...),15);gcf()

