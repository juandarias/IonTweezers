#* Load Julia packages
using LinearAlgebra, DelimitedFiles, BSON

#* Load custom packages
include(srcdir()*"/constants.jl");
using coupling_matrix, crystal_geometry, hamiltonians, analysis

#* Run default functions
