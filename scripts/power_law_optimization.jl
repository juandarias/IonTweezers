using DrWatson, LinearAlgebra, Optim, JLD
@quickactivate "IonTweezers"
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
using coupling_matrix, crystal_geometry

"================"
# Definitions
"================"

Chop_Sort(x) = sort(round.(x,digits=5), dims=2)
DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./maximum(coupling_matrix)
DividebyMin(coupling_matrix::Array{Float64,2}) = coupling_matrix./coupling_matrix[1,size(coupling_matrix,1)]

"========================="
# Crystal parameters
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
PositionIons(Nions,ωtrap,z0,[a,b],plot_position=true)
pos_ions = PositionIons(Nions,ωtrap,z0,[a,b],plot_position=false)
pos_ions=Chop_Sort(pos_ions)
hess_hom=Hessian(pos_ions, ωtrap, z0, [a,b]; planes=[1])

"========================="
# Loop over different power-law decays
"========================="

# Target and experimental matrices and phonon frequencies
JPL(α) = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]
JPLexp(parms) = Jexp1D(pos_ions, vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2])), ωtrap, parms[Nions÷2+1], planes=[1], equidistant=true, size_crystal=z0, coeffs_field=[a,b], hessian=hess_hom)[1:2];

# Seed and constrains
initial_μ = 1.89718;
initial_parms = vcat(0.05*ones(Nions÷2), initial_μ);
Ωmax = 1.0; μmin = 0.5; μmax = 20;
lc = append!(zeros(Nions÷2), μmin);
uc = append!(Ωmax*ones(Nions÷2), μmax);

# Power law coefficients
αs = collect(1:0.25:4);
solution_1 = zeros(7,13);
solution_2 = zeros(7,13);

for n in 1:13
    α=αs[n]

    # Objective functions
    ϵ_J1(parms) = norm(DividebyMax(JPLexp(parms)[1]) - JPL(α))
    ϵ_J2(parms) = 1000*norm(normalize(JPLexp(parms)[1]) - normalize(JPL(α)))

    
    #Optimization
    solutionGD_1 = Optim.optimize(ϵ_J1, lc, uc, initial_parms, Fminbox(GradientDescent()))
    solutionGD_2 = Optim.optimize(ϵ_J2, lc, uc, initial_parms, Fminbox(GradientDescent()))
    solution_1[:,n]= solutionGD_1.minimizer
    solution_2[:,n]= solutionGD_2.minimizer

    println("Running: "*string(α))
    #Collect parameters
    parms_1 = Dict(:ωtrap => ωtrap, :Ωpin => solution_1[:,n][1:Nions÷2], :μnote => solution_1[:,n][Nions÷2 + 1], :axis => "x", :homogeneous => true, :coeffs_field => [a,b], :crystal_size=> z0*1E6, :model => replace("r^(-\$α)", "\$α" => "$α"), :optimizer => Dict(:method => "GradientDescent", :objective => "norm(Jₜ/norm(Jₜ)-Jₑ/norm(Jₑ))"))
    parms_2 = Dict(:ωtrap => ωtrap, :Ωpin => solution_2[:,n][1:Nions÷2], :μnote => solution_2[:,n][Nions÷2 + 1], :axis =>    "x", :homogeneous => true, :coeffs_field => [a,b], :crystal_size=> z0*1E6, :model => replace("r^(-\$α)", "\$α" => "$α"), :optimizer => Dict(:method => "GradientDescent", :objective => "norm(Jₜ/max(Jₜ)-Jₑ/max(Jₑ))"))

    #Save solution and parameters
    parms_1[:Jexp] = JPLexp(solution_1[:,n])[1]; parms_1[:ωm] = JPLexp(solution_1[:,n])[2]
    parms_2[:Jexp] = JPLexp(solution_2[:,n])[1]; parms_2[:ωm] = JPLexp(solution_2[:,n])[2]
    wsave(datadir("PowerLaw", savename(parms_1, "bson")), parms_1)
    wsave(datadir("PowerLaw", savename(parms_2, "bson")), parms_2)
    wsave(datadir("PowerLaw", savename(parms_1, "jld")), "solution", solutionGD_1)
    wsave(datadir("PowerLaw", savename(parms_2, "jld")), "solution", solutionGD_2)

end


