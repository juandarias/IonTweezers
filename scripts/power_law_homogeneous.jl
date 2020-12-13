using DrWatson, PyPlot
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using optimizers, coupling_matrix, crystal_geometry, plotting_functions

"================"
# Definitions
"================"
MHz = 1E6; μm = 1E-6;
Chop_Sort(x) = sort(round.(x,digits=5), dims=2)


"========================="
# Parameters and constants
"========================="

Nions = 12; 
z0 = 40*10^-6; d = 2*z0/Nions; d += d/(Nions -1);


### Fitting F = ax + bx^3 ###
a,b = 1.84*10^-14, -0.000110607

### Frequencies ###
ωtrap = 2*pi*1E6*[0.6, 4.0, 0.2];
ωtrap[3] = abs(√(ee^2/(2π*ϵ0*mYb*d^3)))

"========================="
# Positions and Hessian
"========================="

pos_ions = PositionIons(Nions,ωtrap,z0,[a,b],plot_position=false)
pos_ions=Chop_Sort(pos_ions)

hess=Hessian(pos_ions, ω_trap;planes=[1],[a,b],)


"==============="
# Initial seeds
"==============="

μnote1 = 1.89718;
μnote2 = 2.44607;
μnote3 = 3.04689;
Ωpin1 = sqrt.([0.025135,0.0795449,0.0440598,0.00817738,0.0177739,0.0170886,0.0170886,0.0177739,0.00817738,0.0440598,0.0795449,0.025135]);
Ωpin2 = sqrt.([0.326594,0.11354,0.048535,0.142741,0.197697,0.311001,0.311001,0.197697,0.142741,0.048535,0.11354,0.326594]);
Ωpin3 = sqrt.([0.532216,0.259763,0.0405636,0.03057,0.00945159,0.29059,0.29059,0.00945159,0.03057,0.0405636,0.259763,0.532216]);

PL1 = Dict(:ωtrap => ωtrap, :Ωpin => Array(Ωpin1), :μnote => μnote1, :axis => "x", :homogeneous => true, :coeffs_field => [a,b], :crystal_size=> z0*1E6, :model => "r^(-1)")
PL2 = Dict(:ωtrap => ωtrap, :Ωpin => Array(Ωpin2), :μnote => μnote2, :axis => "x", :homogeneous => true, :coeffs_field => [a,b], :crystal_size=> z0*1E6, :model => "r^(-2)")
PL3 = Dict(:ωtrap => ωtrap, :Ωpin => Array(Ωpin3), :μnote => μnote3, :axis => "x", :homogeneous => true, :coeffs_field => [a,b], :crystal_size=> z0*1E6, :model => "r^(-3)")

JPL3x, λm, bm = Jexp1D(pos_ions, Ωpin3, ωtrap, μnote3, planes=[1], equidistant=true, size_crystal=z0, coeffs_field=[a,b]);
nJPL3x= JPL3x./(JPL3x[Nions,1])



"====================="
# Optimization with GD
"====================="


# Target and experimental matrix
JPL(α) = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]
JPLexp(parms) = Jexp1D(pos_ions, parms[1:Nions], ωtrap, parms[Nions+1], planes=[1], equidistant=true, size_crystal=z0, coeffs_field=[a,b]);


# Parameters and error function
initial_Ω = PL1[:Ωpin]; initial_μ = PL1[:μnote]; initial_parms = vcat(initial_Ω, initial_μ)
ϵ_J(parms, α) = norm((Jhex(parms)./maximum(abs.(Jhex(parms)))) - JPL(α))

# Optimization 
solution = GDPin(ϵ_J, initial_parms, 15.0, 7; μmax = 20.0, c1max = 10.0, c1min=-10.0, show_trace=true)
min_err[w] = solution.minimum  
solns[:,w] = solution.minimizer  
    




#Save parameters and resuls
PL1[:Jexp] = JPL1x
wsave(datadir("Article", savename(PL1, "bson")), PL1)


"======================"
# Plots and output data
"======================"

lnJPL3x =replace!(log.(nJPL3x), -Inf => 0.0)
lnJPL2x =replace!(log.(nJPL2x), -Inf => 0.0)
lnJPL1x =replace!(log.(nJPL1x), -Inf => 0.0)

format_Plot.NormalPlot()
PlotMatrix3D(lnJPL3x, save_results=true, name_figure=savename(PL3)*"log_matrix_plot_normal", location="Article")
gcf()



format_Plot.SmallerPlot()
TweezerStrength2(pos_ions,PL1[:Ωpin], save_results=true, name_figure=savename(PL1)*"tweezer_array", location="Article")
gcf()

