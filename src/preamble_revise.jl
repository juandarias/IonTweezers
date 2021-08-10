#* Load Julia packages
using PyPlot, LinearAlgebra, DelimitedFiles, BSON, PyCall, LineSearches, Optim

#* Load custom packages
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl");
using coupling_matrix, crystal_geometry, analysis, plotting_functions, data_export
using models.general

#* Run default functions
RectangularPlot();