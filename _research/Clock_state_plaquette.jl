#* Load modules
using DrWatson, BSON, LinearAlgebra, Arpack, DelimitedFiles, PyCall, SparseArrays, Revise
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/hamiltonians.jl");
include(srcdir()*"/observables.jl");
using operators_basis


