using DrWatson, PyPlot
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using optimizers, coupling_matrix, crystal_geometry, plotting_functions



"========================="
# Parameters and constants
"========================="

MHz = 1E6; μm = 1E-6;

Nions = 12; ω_trap = 2π*MHz*[2, 0.6, 0.07];
prefactor = ee^2/(4*π*ϵ0);

Chop_Sort(x) = sort(round.(x,digits=5), dims=2)

"========================="
# Positions and Hessian
"========================="

pos_ions = PositionIons(Nions,ω_trap,plot_position=false);
pos_ions=Chop_Sort(pos_ions)

hess=Hessian(pos_ions, ω_trap;planes=[2])


"=================="
# Coupling matrix
"=================="


Hom1D = Dict(:ωtrap => ω_trap, :Ωpin=> Array(sqrt.([8.81055,18.7972,35.5,40.716,48.6586,50.0153,50.0153,48.6586,40.716,35.5,18.7972,8.81055])), :μnote => 7.01356, :axis => "y", :homogeneous => false)

Jhom1D, λm, bm = Jexp1D(pos_ions, Ωpin1D, ω_trap, μ_note, planes=[2])
nJhom1D= Jhom1D./(maximum(abs.(Jhom1D)))

#Save parameters and resuls
Hom1D[:Jexp] = nJhom1D
wsave(datadir("Article", savename(Hom1D, "bson")), Hom1D)

"======================"
# Plots and output data
"======================"


format_Plot.NormalPlot()
PlotMatrixArticle(nJhom1D, save_results=true, name_figure=savename(paramsHom1D)*"normal", location="Article")
gcf()


format_Plot.SmallestPlot()
TweezerStrength2(pos_ions,Hom1D[:Ωpin])
gcf()
