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


Chop_Sort(x) = sort(round.(x,digits=5), dims=2)

"========================="
# Positions and Hessian
"========================="

pos_ions, figcrystal = PositionIons(Nions, ω_trap, plot_position=true, tcool=500E-6, cvel=10E-20)

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


homogeneous_coupling = BSON.load(datadir("Article","axis=y_homogeneous=false_μnote=7.014.bson"))

Ωsol =  homogeneous_coupling[:Ωpin]*ω_trap[3]/(2π*MHz)
μsol = homogeneous_coupling[:μnote]*ω_trap[3]/(2π*MHz)


evals, evecs = eigen(hess)
λ0 = sqrt.(evals)*ω_trap[3]/(2π*MHz)
λsol = sqrt.(Jexp1D(pos_ions, homogeneous_coupling[:Ωpin], ω_trap, homogeneous_coupling[:μnote], hessian=hess)[2])*(ω_trap[3]/(2π*MHz))

ExportModeSpectra(λ0, λsol, "Homogeneous_coupling", "Article")

ExportTikzTweezerGraph(pos_ions, Ωsol, "Homogeneous_coupling", "Article", plane="YZ")

