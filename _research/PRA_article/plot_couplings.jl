
push!(LOAD_PATH, "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Code/Julia/2DCrystal/framework/")
using optimizers, coupling_matrix, constants
using JLD2, PyPlot, LinearAlgebra




"=========================="
# Parameters and input data
"=========================="

    global Nions, Nmodes = 7, 7
    global pos_ions = [[0.611268, 0, -1.27509], [-0.798623, 0, -1.16692], [1.40989, 
    0, -0.108169], [0, 0, 0], [-1.40989, 0, 0.108169], [0.798623, 0, 
    1.16692], [-0.611268, 0, 1.27509]]
    ω_trap = 2*π*[0.2,0.8,0.2] #in MHz

    minSG = jldopen("JSG_test1rep1.jld2")["solns"]
    minSGrep1 = jldopen("JSG_test1_minimizer.jld2")["solns"]
    Jhex(parms) = Jexp(pos_ions, parms[1:Nions], ω_trap, parms[Nions+1])[1]




for i in 3:3
    Jexp = Jhex(minSG[:,i])/norm(Jhex(minSG[:,i]))
    matshow(Jexp)
    colorbar()
    figure()
    colors = vcat([[Jexp[j,i] for i in j+1:Nions] for j in 1:Nions]...)
    quiver(xquiver,zquiver,uquiver,vquiver, colors, angles="xy", scale_units="xy", scale=1, headaxislength=0,headlength=0, headwidth = 1)
    scatter(posx[1],posz[1])
    xlim(-2.5,2.5)
    ylim(-2.5,2.5)
end

    

minSG[:,7]

    xlim(-2.5,2.5)
    ylim(-2.5,2.5)

    Ωmin = string(round.(minSGrep1[:,7][1:7]; digits=2))

    
    Jexpres = Jhex(minSGrep1[:,7])/norm(Jhex(minSGrep1[:,7]))


    res = GraphCoupling(Jexpres, pos_ions,7)
    res = PlotMatrix(Jexpres, JSG,minSGrep1[:,7])

    