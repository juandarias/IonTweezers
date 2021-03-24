using DrWatson, PyPlot, LinearAlgebra
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using optimizers, coupling_matrix, crystal_geometry, plotting_functions


"============="
# Definitions
"============="

    MHz = 1E6; μm = 1E-6;

"========================="
# Auxiliary functions
"========================="

    Chop_Sort(x) = sort(round.(x,digits=5), dims=2)
    Chop_Sort2D(A) = round.(A[:, sortperm(A[3,:])],digits=5)
    Σ = sum
    ⊗(A,B) = kron(A',B)


    function NearestNeighbour(positions, cutoff)
        number_ions = size(positions)[2]
        indecesNN = []
        for m=1:number_ions
            dₘ = [norm(positions[:,m]-positions[:,n]) for n=1:number_ions]
            indexNN = findall(d -> d < cutoff && d != 0, dₘ)
            push!(indecesNN, indexNN)
        end
        return indecesNN
    end

    function TriangularLattice(d₀)
        triang_lattice = zeros(3,19)

        d₁ = d₀*cos(π/6)
        for n=1:6
            triang_lattice[:,n+1] = [0.0, d₀*cos(n*π/3), d₀*sin(n*π/3)] #depth=1
        end
        for n=1:6
            triang_lattice[:,2*n+7] = [0.0, 2*d₀*cos(n*π/3), 2*d₀*sin(n*π/3)] #depth =2; vertices
        end
        for n=1:6
            triang_lattice[:,2*n-1+7] = [0.0, 2*d₁*cos(n*π/3-π/6), 2*d₁*sin(n*π/3-π/6)] #depth =2; middle points
        end


        return triang_lattice
    end


    function TriangularLattice(d₀, depth)
        d₁ = d₀*cos(π/6)
        
        function DisplaceYZ(unit_cell)
            left_cell = unit_cell + [0 0 0; -d₀/2 -d₀/2 -d₀/2; d₁ d₁ d₁]
            right_cell = unit_cell + [0 0 0; d₀/2 d₀/2 d₀/2; d₁ d₁ d₁]
            return left_cell, right_cell
        end

        u0 = [0 0 0; 0 -d₀/2 d₁; 0 d₀/2 d₁]' #first unit cell depth 0

        #* build (north) unit cell up to target depth
        ud = []
        push!(ud, u0)
        for d = 0:depth-2
            for i = length(ud)-d:length(ud)
                uleft, uright = DisplaceYZ(ud[i])
                push!(ud, uleft)
                push!(ud, uright)
            end
            unique!(ud)
        end
        ud_a = unique(hcat(ud...),dims=2)

        #* Rotate unit cell to obtain complete crystal

        R(n)=[0 0 0; 0 cos(n*π/3) sin(n*π/3); 0 cos(n*π/3+π/2) sin(n*π/3+π/2)] #Rotation matrix

        ut = []
        for n=0:5
            push!(ut, R(n)*ud_a)
        end

        #* filter repeated coordinates
        positions_ions = replace(round.(hcat(ut...), digits=6), -0.0 => 0.0) #some ridiculous filtering
        positions_ions = unique(positions_ions, dims=2)

        #* sort by distance and angle
        positions_sorted = sortslices(positions_ions, dims=2, by = x -> norm(x))
        
        for d in 1:depth
            total_num_vertices = 6*Int(d*(d+1)/2)+1;
            first_index_level = total_num_vertices - 6*d + 1;
            positions_sorted[:,first_index_level:total_num_vertices] = sortslices(positions_sorted[:,first_index_level:total_num_vertices], dims=2, by = x -> atan(x[2],x[3]))
        end
        return positions_sorted
    end


    function Ωhex(Ωsol)
        depth = (length(Ωsol)-1) ÷ 2
        number_vertices = 6*Int(depth*(depth+1)/2)+1;
        Ωpin = zeros(number_vertices)
        Ωpin[1] = Ωsol[1]
        Ωpin[2:7] .= Ωsol[2]
        Ωpin[9:2:19] .= Ωsol[3]
        Ωpin[8:2:18] .= Ωsol[4]
        if depth == 3
            Ωpin[22:3:37] .= Ωsol[5]
            Ωpin[20:3:35] .= Ωsol[5]
            Ωpin[21:3:36] .= Ωsol[6]
        end
        return Ωpin
    end


"==========================="
# Parameters and constants
"==========================="
    
    depth = 3;
    Nions = 6*Int(depth*(depth+1)/2)+1; 
    ### Frequencies ###
    ωtrap = 0.8*2π*MHz*[3.0, 0.2, 0.2]; #19 ions
    ωtrap = 0.7*2π*MHz*[3.0, 0.2, 0.2]; #37 ions

"========================="
# Positions and Hessian
"========================="

    pos_ions_even = TriangularLattice(1.5, 3)
    Crystal2D(pos_ions_even, ones(37); plane="YZ");gcf()

    #pos_ions, figcrystal = PositionIons(Nions, ωtrap, plot_position=true, tcool=500E-6, cvel=10E-20)
    pos_ions_uneven, figcrystal_seed = PositionIons(Nions, ωtrap, seed=pos_ions_even, plot_position=true, tcool=500E-6, cvel=10E-20)

    hess_X = Hessian(pos_ions, ωtrap; planes=[1])# Out-of-plane modes
    hess_XYZ = Hessian(pos_ions, ωtrap; planes=[1,2,3])# In plane modes

    hess_X_even = Hessian(pos_ions_even, ωtrap; planes=[1])# Out-of-plane modes
    hess_XYZ_even = Hessian(pos_ions_even, ωtrap; planes=[1,2,3]);

    hess_X_uneven = Hessian(pos_ions_uneven, ωtrap; planes=[1]);# Out-of-plane modes
    hess_XYZ_uneven = Hessian(pos_ions_uneven, ωtrap; planes=[1,2,3]);# In plane modes


"========================="
# Target and experimental matrices
"========================="

    # Target and experimental matrices and phonon frequencies
    iNN = NearestNeighbour(pos_ions_uneven, 1.6)
    JAFM_NN =zeros(Nions, Nions)
    for i in 1:Nions, j in iNN[i]
        JAFM_NN[i,j] = 1
    end

    GraphCoupling(JAFM_NN, pos_ions_uneven, plane="YZ");gcf()

    JexpOOP(parms) = Jexp1D(pos_ions_even, Ωhex(parms), ωtrap, parms[2*depth+1], hessian=hess_X_even)[1];
    JexpOOP_uneven(parms) = Jexp1D(pos_ions_uneven, Ωhex(parms), ωtrap, parms[2*depth+1], hessian=hess_X_uneven)[1];
    #JexpIPY(parms) = Jexp2D(pos_ions_seed_l, ΩhexIP(parms), ωtrap, parms[5], hessian=hessYZ_seed, kvec=[0,1,0])[1]; #Coupling only along Y-direction
    #JexpIPZ(parms) = Jexp2D(pos_ions_seed_l, ΩhexIP(parms), ωtrap, parms[5], hessian=hessYZ_seed, kvec=[0,0,1])[1]; #Coupling only along Z-direction
    #JexpIPYZ(parms) = Jexp2D(pos_ions_seed_l, ΩhexIP(parms), ωtrap, parms[5], hessian=hessYZ_seed, kvec=[0,1,1])[1]; #Coupling only along YZ-direction

    λp(parms) = Jexp1D(pos_ions_even, Ωhex(parms), ωtrap, parms[2*depth+1], hessian=hess_X_even)[2];

"========================="
# Objective function and constrains
"========================="

    # Seed and constrains
    initial_μ = 10.0;
    initial_parms = vcat(0.05*ones(2*depth), initial_μ);
    Ωmax = 20.0;
    μmin = 0.05;
    μmax_OOP = 20;
    #μmax_IP = 300;
    lc = append!(zeros(2*depth), μmin);
    uc_OOP = append!(Ωmax*ones(2*depth), μmax_OOP);
    #uc_IP = append!(Ωmax*ones(4), μmax_IP);

    # Objective functions
    DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./maximum(abs.(coupling_matrix))
    ϵ_JOOP(parms) = norm(DividebyMax(JexpOOP(parms)) - JAFM_NN);
    ϵ_JOOP_uneven(parms) = norm(DividebyMax(JexpOOP_uneven(parms)) - JAFM_NN);
    #ϵ_JIPY(parms) = norm(DividebyMax(JexpIPY(parms)) - JAFM_NN)
    #ϵ_JIPZ(parms) = norm(DividebyMax(JexpIPZ(parms)) - JAFM_NN)
    #ϵ_JIPYZ(parms) = norm(DividebyMax(JexpIPYZ(parms)) - JAFM_NN)


"========================="
# Optimization
"========================="

    solution_nogradOOP_even = Optim.optimize(ϵ_JOOP, lc, uc_OOP, initial_parms, Fminbox(LBFGS(linesearch=BackTracking(order=3))), Optim.Options(time_limit = 500.0, store_trace=true))

    seed_even = solution_nogradOOP_even.minimizer
    solution_nogradOOP_uneven = Optim.optimize(ϵ_JOOP_uneven, lc, uc_OOP, seed_even, Fminbox(LBFGS(linesearch=BackTracking(order=3))), Optim.Options(time_limit = 500.0, store_trace=true))
    
    #solution_nogradOOP_seed_CG = Optim.optimize(ϵ_JOOP, lc, uc_OOP, initial_parms, Fminbox(ConjugateGradient(linesearch = BackTracking(order=3))), Optim.Options(time_limit = 500.0, store_trace=true))

    #solution_nogradIPY = Optim.optimize(ϵ_JIPY, lc, uc_IP, initial_parms, Fminbox(LBFGS(linesearch=BackTracking())), Optim.Options(time_limit = 500.0, store_trace=true))

    #solution_nogradIPZ = Optim.optimize(ϵ_JIPZ, lc, uc_IP, initial_parms, Fminbox(LBFGS(linesearch=BackTracking())), Optim.Options(time_limit = 500.0, store_trace=true))

    #solution_nogradIPYZ = Optim.optimize(ϵ_JIPYZ, lc, uc_IP, initial_parms, Fminbox(LBFGS(linesearch=BackTracking())), Optim.Options(time_limit = 500.0, store_trace=true))


"===================================================="
# Eigenvalues and eigenstates with transversal field
"===================================================="

    using Arpack: eigs

    Jsol_even = JsolOOP_N[1:19,1:19]
    Jsol_uneven = JsolOOP_uneven_N[1:19,1:19]
    JAFM_t = JAFM_NN[1:19,1:19]


    λTI = zeros(8,11)
    ϕTI = zeros(2^19,8,11)
    for h=0:10
        h0 = h/10;
        field = h0*ones(19);
        HIsing = TransversalIsing(Jsol_uneven, field)
        λ, ϕ = eigs(HIsing, nev=8, which=:SR)#, tol=1E-5)
        λTI[:,h+1] = λ
        ϕTI[:,:,h+1] = ϕ
    end

    function MagnetizationZ(state)
        N = Int(log(length(state))/log(2))
        M = sum([state'*σᶻᵢ(i,N)*state for i=1:N])
        return M
    end

    fig, ax = plt.subplots()
    for h=1:10
        scatter(h/10*ones(8), λTI_ideal[:,h+1],marker="_");
    end
    xlabel("\$h/\\text{max} (J)\$")
    gcf()

    Mz_ideal = zeros(8,11)
    for m=1:8, h=1:11
        Mz_ideal[m,h] = MagnetizationZ(ϕTI_ideal[:,m,h])
    end




"================="
# Plots of solutions
"================="


    #* Even results
    JsolOOP_N = DividebyMax(JexpOOP(solution_nogradOOP_even.minimizer))
    λOOP = sqrt.(λp(solution_nogradOOP_even.minimizer))*(ωtrap[3]/(2π*MHz))
    μsolOOP = solution_nogradOOP_even.minimizer[end]*ωtrap[3]/(2π*MHz)
    λ0OOP = sqrt.(eigen(hess_X_even).values)*(ωtrap[3]/(2π*MHz))

    #* Uneven results
    JsolOOP_uneven_N = DividebyMax(JexpOOP_uneven(solution_nogradOOP_uneven.minimizer))
    λOOP = sqrt.(λp(solution_nogradOOP_even.minimizer))*(ωtrap[3]/(2π*MHz))
    μsolOOP = solution_nogradOOP_even.minimizer[end]*ωtrap[3]/(2π*MHz)
    λ0OOP = sqrt.(eigen(hess_X_even).values)*(ωtrap[3]/(2π*MHz))


    format_Plot.NormalPlot()

    GraphCoupling(JsolOOP_N, pos_ions_even; plane="YZ"); savefig(plotsdir("TriangularLattice", "19_ions_OOP_uneven_AFM_all.png"));gcf() #All
    GraphCoupling(JsolOOP_uneven_N, pos_ions_uneven; plane="YZ", upper_cutoff=1.55, lower_cutoff=1.0, zero_offset=0.002);savefig(plotsdir("TriangularLattice", "37_ions_OOP_uneven_AFM_NN.png"));gcf() #NN
    GraphCoupling(JsolOOP_N, pos_ions_even; plane="YZ", upper_cutoff=10.0, lower_cutoff=1.6, zero_offset=0.002);gcf()
     #savefig(plotsdir("TriangularLattice", "19_ions_OOP_uneven_AFM_NNN.png")); gcf() #NNN

    function CouplingErrors(target_matrix, result_matrix)
        dims = size(target_matrix)[2]
        target_couplings = result_matrix - result_matrix.*(iszero.(target_matrix))
        residual_couplings = result_matrix.*(iszero.(target_matrix))
        error_target_couplings = [target_couplings[i,j]!=0 ? 1-target_couplings[i,j] : 0 for i in 1:dims, j in 1:dims]
        return error_target_couplings, residual_couplings
    end
    
    ϵ_target_even, ϵ_residual_even = CouplingErrors(JAFM_NN, JsolOOP_N)
    ϵ_target_uneven, ϵ_residual_uneven = CouplingErrors(JAFM_NN, JsolOOP_uneven_N)
        
    GraphCouplingError(ϵ_target_even, pos_ions_even; plane="YZ", upper_cutoff=1.6, lower_cutoff=1.0, zero_offset=0.002);savefig(plotsdir("TriangularLattice", "37_ions_even_OOP_error_coupling_NN.png"));gcf() #NN
    GraphCouplingError(abs.(ϵ_residual_even), pos_ions_even; plane="YZ", upper_cutoff=7.0, lower_cutoff=1.6, zero_offset=0.01);savefig(plotsdir("TriangularLattice", "37_ions_even_OOP_error_coupling_NNN.png"));gcf()


    fig, ax = plt.subplots()
    Crystal2D(pos_ions_seed_l, Ωhex(solution_nogradOOP_seed_l.minimizer), plane="YZ", offset_label=0.1);savefig(plotsdir("TriangularLattice", "19_ions_OOP_uneven_AFM_solution.png"));gcf()

    PlotSpectraSingle(λOOP, μsolOOP; unpinned_frequencies=λ0OOP);savefig(plotsdir("TriangularLattice", "19_ions_OOP_uneven_AFM_spectra.png"));gcf()

    b0_uneven = eigen(hess_X_seed).vectors[:,1]
    b0_even = eigen(hess_X_tri).vectors[:,1]


    ModeStrength(pos_ions_seed_l, b0_uneven; plane="YZ"); gcf()
    savefig(plotsdir("TriangularLattice", "19_ions_uneven_first_mode_OOP.png"));gcf()

    ModeStrength(pos_ions_tri, b0_even; plane="YZ"); savefig(plotsdir("TriangularLattice", "19_ions_even_first_mode_OOP.png"));gcf()

    pos_ions_tri = TriangularLattice(1.4)
    fig,ax =plt.subplots()
    scatter(pos_ions[1][2,:],pos_ions[1][3,:]);scatter(pos_ions_tri[2,:],pos_ions_tri[3,:]);gcf()

    
"================="
# Export Results
"================="

    note_Triangular_AFM = "Regular crystal (dₘ ~ 11.28 μm). Considering OOP modes (X axis), and hexagonal symmetry of tweezers. Gradient is numerical"

    dict_Triangular_AFM = SummaryOptimization(ωtrap, pos_ions_seed_l, hess_X_seed, "TriangularLattice_AFM", "||Jₑ/max(Jₑ)-Jₜ||₂", solution_nogradOOP_seed_l, Ωhex(solution_nogradOOP_seed_l.minimizer), solution_nogradOOP_seed_l.minimizer[end], JsolOOP_N, note=note_Triangular_AFM)

    wsave(datadir("TriangularLattice", string(Dates.today())*"_Triangular_lattice_even_AFM.bson"), summary_dict)

    ExportTikzCouplingGraph(pos_ions_seed_l, JAFM_NN, JsolOOP_N, "19_ions_AFM_triangular", "TriangularLattice")

    ExportTikzTweezerGraph(position_ions_AFM_uneven, Ωpin_AFM_uneven*ωtrap[3]/(2π*MHz), "Triangular_lattice_even_AFM", "TriangularLattice")

"================="
# Import Results
"================="

    using BSON
    results_triangular_even = BSON.load(datadir("TriangularLattice","2021-03-03_Triangular_lattice_even_AFM.bson"))
    results_triangular_uneven = BSON.load(datadir("TriangularLattice","2021-03-03_Triangular_lattice_uneven_AFM.bson"))

    ωtrap = results_triangular_uneven[:trap_frequency]
    μsol = results_triangular_uneven[:beatnote]*ωtrap[3]
    Ωsol_uneven = results_triangular_uneven[:pinning_frequency]*ωtrap[3]
    hess_X_uneven = results_triangular_uneven[:hessian]
    Jres = results_triangular_uneven[:result_matrix]
    pos_ions_uneven = results_triangular_uneven[:position_ions]

    GraphCoupling(Jres, pos_ions_uneven, plane="YZ", upper_cutoff=1.6, lower_cutoff=1.0, zero_offset=0.002);gcf() #NN


    ϵ_target, ϵ_residual = CouplingErrors(JAFM_NN, Jres)


    ExportTikzCouplingGraphError(pos_ions_uneven, JAFM_NN, abs.(ϵ_target), abs.(ϵ_residual), "Triangular_lattice_AFM_19ions_error", "Article", threshold=0.02)

    GraphCouplingError(ϵ_target, pos_ions_even; plane="YZ", upper_cutoff=1.6, lower_cutoff=1.0, zero_offset=0.002);gcf()
    savefig(plotsdir("TriangularLattice", "19_ions_uneven_OOP_error_coupling_NN.png"));gcf() #NN
    GraphCouplingError(abs.(ϵ_residual), pos_ions_even; plane="YZ", upper_cutoff=10.0, lower_cutoff=1.6, zero_offset=0.005);gcf()
    savefig(plotsdir("TriangularLattice", "19_ions_uneven_OOP_error_coupling_NNN.png")); gcf()

    writedlm(datadir("TriangularLattice", "AFM_Triangular_Coupling_Matrix.dat"), Jres)
    writedlm(datadir("TriangularLattice", "37_ions_AFM_Triangular_uneven_evals.dat"), λTI)
    writedlm(datadir("TriangularLattice", "37_ions_AFM_Triangular_uneven_evec.dat"), ϕTI)
    writedlm(datadir("TriangularLattice", "37_ions_AFM_Triangular_uneven_magnetization.dat"), Mz)



    ExportTikzTweezerGraph(position_ions_AFM_uneven, Ωpin_AFM_uneven*ωtrap[3]/(2π*MHz), "Triangular_lattice_even_AFM", "TriangularLattice")


