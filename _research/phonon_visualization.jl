push!(LOAD_PATH, "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Code/Julia/2DCrystal/framework/")
push!(LOAD_PATH, "/Users/gebruiker/Dropbox/Projects/QuantumSimulationPhD/Code/Julia/2DCrystal/framework/")
push!(LOAD_PATH, "C:/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Code/Julia/2DCrystal/framework/")

base_directory = "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Code/Julia/2DCrystal/"
#base_directory = "/Users/gebruiker/Dropbox/Projects/QuantumSimulationPhD/Code/Julia/2DCrystal/"
output_location = base_directory*"/tests/Jstar/output/"
push!(LOAD_PATH, base_directory*"/framework")
push!(LOAD_PATH, base_directory*"/sandbox")

using LinearAlgebra, SharedArrays#, Plotly #, Nabla, NLopt
using coupling_matrix, constants, plotting_functions



"=========================="
# Notes
# -review which of the modes are actually contributing to the desired interaction, as it appears to be that not only one mode is responsible of the star interaction. Check the amplitudes of the dominating modes.
# -check if the calculation of the coupling matrix is oke!
# the order of magnitude of the beatnote and the amplitudes of the mode are critical.
"=========================="



"=========================="
# Parameters and input data
"=========================="

    global Nions = 7
    global Nmodes = 7
    #global pos_ions = [[0.611268, 0, -1.27509], [-0.798623, 0, -1.16692], [1.40989, 0, -0.108169], [0, 0, 0], [-1.40989, 0, 0.108169], [0.798623, 0, 1.16692], [-0.611268, 0, 1.27509]]
    #ω_trap = 2*π*[0.2,0.8,0.2] #in MHz

    global pos_ions = [[-0.474753, 0, -1.33195], [0.916129, 0, -1.07713], [-1.39088, 0, -0.254829], [0, 0, 0], [1.39088, 0, 0.254829], [-0.916129, 0, 1.07713], [0.474753, 0, 1.33195]] # ω_trap = 2*π*[0.2,0.5,0.2]
    ω_trap = 2*π*[0.2,0.5,0.2]


using PyPlot

"==================="
# Define functions
"==================="

    #### Star interaction
    vfmS = [[4,1],[4,2],[4,3],[4,5],[4,6],[4,7]]
    Jstar = Jtarget(vfmS,[],Nions)
    
    #### Star interaction matrix
    Jhex(Ω, μ) = Jexp(pos_ions, Ω, ω_trap, μ)[1]
    ωhex(Ω) = Jexp(pos_ions, Ω, ω_trap, 10.0)[2]
    bhex(Ω) = Jexp(pos_ions, Ω, ω_trap, 10.0)[3]
    μramanFM(Ω) = real(ωhex(Ω)[21] + 2*π*0.02)
    μramanAFM(Ω) = real(ωhex(Ω)[21] - 2*π*0.02)
    JstarAFM(Ω) = Jhex(Ω, μramanAFM(Ω))
    JstarFM(Ω) = Jhex(Ω, μramanFM(Ω))


    format_Plot.RectangularPlot()
    PlotSpectraArticle(ωhex(zeros(7)),0.0)

    GraphCoupling2(JstarAFM(zeros(7))./minimum(JstarAFM(zeros(7))), pos_ions)

    format_Plot.SquareSmallerPlot()
    PlotMatrixArticle(JstarAFM(zeros(7)))
    
    vfmS = [[4,1],[4,2],[4,3],[4,5],[4,6],[4,7]]
    Jstar = Jtarget(vfmS,[],Nions)
    PlotMatrixArticle(Array(Jstar))

    ΩFM = 2*pi*2
    ΩAFM = 2*pi*10
    Ω_pin_FM = [1e-5, 1e-5, 1e-5, ω_trap[3], 1e-5, 1e-5, 1e-5]
    Ω_pin_AFM = [15, 15, 15, 1e-5, 15, 15, 15]
  
    #### Plot amplitudes

    xpos = [pos_ions[i][1] for i in 1:Nions]
    zpos = [pos_ions[i][3] for i in 1:Nions]



    PyPlot.scatter(xpos,zpos,c=b_starAFM,s=100)
    
    Plotly.plot([Plotly.scatter(x=xpos,y=zpos,mode="markers+text",marker_color=b_starAFM,marker_size=50,text=round.(b_starAFM;digits=3),textposition="bottom center")])

    n = 0
    dataFMp = []
    for i in 15:21
        global n += 10
        phmode = bhex([1,0,0,0,1,1,0]*2*pi)[:,i][8:14]
        freq = round(real(ωhex([1,0,0,0,1,1,0]*2*pi)[i]);digits=3)
        xplot = xpos .+ n
        yplot = zpos 
        append!(dataFMp,[scatter(;x=xplot,y=yplot,mode="markers+text",marker_color=phmode,marker_size=20,text=round.(phmode;digits=3),textposition="bottom center",name=freq,title=string(freq))])
    end

    plot([dataFMp[i] for i in 1:7])





    plot([heatmap(z=b_starAFM*b_starAFM')])
    

    n = 0
    dataAFM = []
    for i in 15:21
        global n += 10
        phmode = bhex(Ω_pin_AFM)[:,i][8:14]
        freq = round(ωhex(Ω_pin_AFM)[i];digits=3)
        xplot = xpos .+ n
        yplot = zpos 
        append!(dataAFM,[scatter(;x=xplot,y=yplot,mode="markers+text",marker_color=phmode,marker_size=20,text=round.(phmode;digits=3),textposition="bottom center",name=freq)])
    end

    plot([dataAFM[i] for i in 1:7])
