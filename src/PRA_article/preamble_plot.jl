#* Load Julia packages
using PyPlot, LinearAlgebra, DelimitedFiles, BSON, PyCall

#* Load custom packages
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using coupling_matrix, crystal_geometry, hamiltonians, analysis, plotting_functions, data_export

#* Run default function
RectangularPlot();