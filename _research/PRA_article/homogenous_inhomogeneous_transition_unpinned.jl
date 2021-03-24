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
ωtrap = 2π*MHz*[1.0, 1.0, 0.1];

"========================="
# Positions and Hessian
"========================="

pos_ions = PositionIons(Nions, ωtrap, plot_position=false, tcool=500E-6, cvel=10E-20)
pos_ions = Chop_Sort(pos_ions)


function PositionEvolution(ref_positions, χ)
    num_ions = size(ref_positions)[2]
    new_positions = zero(ref_positions)
    if num_ions % 2 == 0
        d₀ = ref_positions[3,num_ions÷2] - ref_positions[3,num_ions÷2 + 1]
        dₙ = [ref_positions[3,i]-ref_positions[3,i+1] for i=num_ions÷2:num_ions-1]
        Δₙ = dₙ .- d₀
        dₙ̃ = (1-χ)*Δₙ .+ d₀
        xₙ = [ref_positions[3,num_ions÷2 + 1] + sum([dₙ̃[n] for n=1:m]) for m=1:num_ions÷2]
        new_positions[3,:] = prepend!(-xₙ, reverse(xₙ))
    elseif num_ions % 2 == 1
        d₀ = ref_positions[3,(num_ions+1)÷2] - ref_positions[3,(num_ions+1)÷2 + 1]
        dₙ = [ref_positions[3,i]-ref_positions[3,i+1] for i=(num_ions+1)÷2:num_ions-1]
        Δₙ = dₙ .- d₀
        dₙ̃ = (1-χ)*Δₙ .+ d₀
        xₙ = [0.0 + sum([dₙ̃[n] for n=1:m]) for m=1:(num_ions-1)÷2]
        new_positions[3,:] = [-xₙ..., 0, reverse(xₙ)...]
    end
    return new_positions
end

Hessian_X(pos) = Hessian(pos, ωtrap; planes=[1])
Jexp(parms) = Jexp1D(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+1], planes=[1], hessian=hessian)[1];
Ωpin(parms) = vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2]))

"==================="
# Effect of spacing
"==================="

function J₀(pos_initial, χ, δ; red = true)
    parms_nopinning = zeros(Nions÷2+1)
    pos_corrected = PositionEvolution(pos_initial, χ)
    hessχ = Hessian_X(PositionEvolution(pos_corrected, χ));
    λm, bm = eigen(hessχ)
    red == true && (μ = √λm[1] - δ)
    red == false && (μ = √λm[end] + δ)
    parms_nopinning[2:2:6] .= 5
    parms_nopinning[7] = μ
    Jexp = Jexp1D(pos_corrected, Ωpin(parms_nopinning), ωtrap, parms_nopinning[Nions÷2+1], planes=[1], hessian=hessχ)[1]
    return Jexp
end

Jexp_red_pinned = zeros(Nions, Nions, 11);
Jexp_blue_pinned = zeros(Nions, Nions, 11);

for n=1:11
    Jexp_red_pinned[:,:,n] = J₀(pos_ions, χs[n], 2π*0.1)
    Jexp_blue_pinned[:,:,n] = J₀(pos_ions, χs[n], 2π*0.1; red=false)
end

PlotArrayMatrices(Jexp_blue_pinned)
gcf()

PlotArrayMatrices(Jexp_red_pinned)
gcf()


"======================="
# Effect of number ions
"======================="

DividebyMaxAbs(coupling_matrix::Array{Float64,2}) = coupling_matrix./maximum(abs.(coupling_matrix))

function J₀ₙ(Nions, χ, δ; red = true)
    parms_nopinning = zeros(Nions+1)
    pos_initial = Chop_Sort(PositionIons(Nions, ωtrap, plot_position=false, tcool=500E-6, cvel=10E-20))
    pos_corrected = PositionEvolution(pos_initial, χ)
    hessχ = Hessian_X(pos_corrected);
    λm, bm = eigen(hessχ)
    red == true && (μ = √λm[1] - δ)
    red == false && (μ = √λm[end] + δ)
    parms_nopinning[Nions+1] = μ
    Jexp = Jexp1D(pos_corrected, parms_nopinning[1:Nions], ωtrap, parms_nopinning[Nions+1], hessian=hessχ, planes=[1])[1]
    return Jexp
end

Jexp_red_Nions = [];
Jexp_blue_Nions = [];
errors = [];
for n=7:16
    try
        push!(Jexp_red_Nions, J₀ₙ(n, 0.0, 2π*0.1))
        push!(Jexp_blue_Nions, J₀ₙ(n, 0.0, 2π*0.1, red=false))
    catch error
        push!(errors, n)
    end
end

PlotArrayMatricesIons(Jexp_red_Nions)


function PlotArrayMatrices(interaction_matrix)
    fig = plt.figure()
    gs = fig.add_gridspec(4,3, hspace=0.25, wspace=-0.5)
    axs = gs.subplots(sharex=true, sharey=true)
    k=0
    for m=1:4, n=1:3
        k+=1
        k!=12 && (mat = axs[m,n].matshow(DividebyMaxAbs(interaction_matrix[:,:,k]), cmap="seismic");mat.set_clim(vmin=-1,vmax=1);axs[m,n].xaxis.tick_bottom();axs[m,n].set_title("\$\\chi=\$"*string(χs[k])))
        k==12 && (axs[m,n].remove())
    #cbar = fig.colorbar(mat)
    end
end

function PlotArrayMatricesIons(interaction_matrix)
    fig = plt.figure()
    gs = fig.add_gridspec(4,3, hspace=0.4, wspace=-0.5)
    axs = gs.subplots(sharex=false, sharey=false)
    k=0
    for m=1:4, n=1:3
        k+=1
        k ∈ [11,12] ? (axs[m,n].remove()) : (mat = axs[m,n].matshow(DividebyMaxAbs(interaction_matrix[k]), cmap="seismic");mat.set_clim(vmin=-1,vmax=1);axs[m,n].xaxis.tick_bottom();axs[m,n].set_title("\$N=\$"*string(k+6)))
    #cbar = fig.colorbar(mat)
    end
end


PlotArrayMatricesIons(Jexp_blue_Nions); savefig(plotsdir("HomtoInhom","no_pinning_size_crystal_blue.png"))