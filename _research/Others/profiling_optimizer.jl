### Test
base_directory =  "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Code/Julia/2DCrystal"
push!(LOAD_PATH, base_directory*"/framework")


using LinearAlgebra, SharedArrays, Optim, JLD2, ProfileView, Profile #, Nabla, NLopt
using optimizers, coupling_matrix, constants
Profile.init(n = 10^7, delay = 0.01)



"===================="
# Global definitions
"===================="
    #
    test_name = "Jstar_radial_test2"
    output_location = base_directory*"/tests/Jstar/output/"

"=========================="
# Parameters and input data
"=========================="
    #
    global Nions, Nmodes = 7, 7;
    global pos_ions = [[0.611268, 0, -1.27509], [-0.798623, 0, -1.16692], [1.40989, 
    0, -0.108169], [0, 0, 0], [-1.40989, 0, 0.108169], [0.798623, 0, 
    1.16692], [-0.611268, 0, 1.27509]];
    ω_trap = 2*π*[0.2,0.8,0.2]; #in MHz
    

"============================================="
# Gradient Descent with numerical derivatives
"============================================="

        #### Star interaction
    vfmS = [[4,1],[4,2],[4,3],[4,5],[4,6],[4,7]]
    Jstar = Jtarget(vfmS,[],Nions)
    Jstar = Jstar/norm(Jstar)        
    Jhex(parms) = JexpOpt(pos_ions, parms[1:Nions], ω_trap, parms[Nions+1],kvec=[1,0,1])[1]
    initial_Ω = zeros(Nions) .+ 0.1
    initial_μ, initial_c1 = 15.0, 10.0 #frequencies in MHz
    initial_parms = vcat(initial_Ω, [initial_μ, initial_c1])
    ϵ_J(parms) = norm(Jhex(parms)/norm(Jhex(parms)) - Array(Jstar) * parms[Nions+2])
    
    CGDPin6(ϵ_J, [0.0001,0001.0001,0001.0001,2.0001,0001.0001,0001.0001,0001.0001,10.0, 1.0];show_trace=true)
    
    @profview GDPin4(ϵ_J, [0.0001,0001.0001,0001.0001,10.0001,0001.0001,0001.0001,0001.0001,10.0, 1.0], 30.0, 7, μmax = 40.0, c1max = 10.0, c1min=-10.0)
    
    Profile.clear()
    Profile.print()
    



    

