#* Load modules
    using DrWatson
    push!(LOAD_PATH, srcdir());
    push!(LOAD_PATH, scriptsdir());

    include(srcdir("preamble_revise.jl"))
    using models.Kondo
        

#* Definitions
    MHz = 1E6; μm = 1E-6;

#* Auxiliary functions

    Chop_Sort(x) = sort(round.(x,digits=5), dims=2)
    Chop_Sort2D(A) = round.(A[:, sortperm(A[3,:])],digits=5)
    Σ = sum
    ⊗(A,B) = kron(A',B)

#* Parameters

    depth = 3;
    d₀ = 1.5; #even spacing
    Nions = 6*Int(depth*(depth+1)/2)+1; 
    
    ### Frequencies ###
    ωₜ = 0.8*2π*MHz*[3.0, 0.2, 0.2]; #19 ions
    ωₜ = 0.7*2π*MHz*[3.0, 0.2, 0.2]; #37 ions

#* Positions and Hessian
    
    pos_ions_even = TriangularLattice(d₀, depth)
    Crystal2D(pos_ions_even, ones(37); plane="YZ");gcf()

    #pos_ions, figcrystal = PositionIons(Nions, ωₜ, plot_position=true, tcool=500E-6, cvel=10E-20)
    pos_ions_uneven, figcrystal_seed = PositionIons(Nions, ωₜ, seed=pos_ions_even, plot_position=true, tcool=500E-6, cvel=10E-20)

    hess_X_even = Hessian(pos_ions_even, ωₜ; planes=[1]);# Out-of-plane modes
    #hess_XYZ_even = Hessian(pos_ions_even, ωₜ; planes=[1,2,3]);

    hess_X_uneven = Hessian(pos_ions_uneven, ωₜ; planes=[1]);# Out-of-plane modes
    #hess_XYZ_uneven = Hessian(pos_ions_uneven, ωₜ; planes=[1,2,3]);


#* Target matrix

    iNN = NearestNeighbour(pos_ions_uneven, 1.6)
    
    JAFM_NN =zeros(Nions, Nions); #target matrix
    for i in 1:Nions, j in iNN[i]
        JAFM_NN[i,j] = 1
    end

#* Experimental matrices

    Jexp(parms::Array{Float64,1}) = Jexp1D(pos_ions_even, parms[1:Nions], ωₜ, parms[Nions+1], hessian=hess_X_even)[1];
    Jexp_uneven(parms::Array{Float64,1}) = Jexp1D(pos_ions_uneven, parms[1:Nions], ωₜ, parms[Nions+1], hessian=hess_X_uneven)[1];

#* Seed and constraints

    #! considering symmetry of lattice and couplings
    μ₀ = 10.0;
    s₀ = vcat(0.05*ones(2*depth), μ₀); #{Ωᵢ, μ}
    Ωmax = 20.0;
    μmin = 0.05;
    μmax = 20;
    lc = append!(zeros(2*depth), μmin);
    uc_OOP = append!(Ωmax*ones(2*depth), μmax);

    function Ωsymm(parms) #rotates results of unit cell to cover all crystal
        depth = (length(parms)-1) ÷ 2
        number_ions = 6*Int(depth*(depth+1)/2)+1;
        Ωpin = zeros(number_ions+1)
        Ωpin[1] = parms[1]
        Ωpin[2:7] .= parms[2]
        Ωpin[9:2:19] .= parms[3]
        Ωpin[8:2:18] .= parms[4]
        if depth == 3
            Ωpin[22:3:37] .= parms[5]
            Ωpin[20:3:35] .= parms[5]
            Ωpin[21:3:36] .= parms[6]
        end
        return vcat(Ωpin, parms[end])
    end

#* Objective functions

    DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./maximum(abs.(coupling_matrix))
    ϵ_J(parms::Array{Float64,1}) = norm(DividebyMax(Jexp(Ωsymm(parms))) - JAFM_NN);
    ϵ_J_uneven(parms::Array{Float64,1}) = norm(DividebyMax(Jexp_uneven(Ωsymm(parms))) - JAFM_NN);

#* Optimization

    algorithm = Fminbox(LBFGS(linesearch=BackTracking(order=3));
    options_solver = Optim.Options(time_limit = 500.0, store_trace=false);

    solution_even = Optim.optimize(ϵ_JOOP, lc, uc_OOP, initial_parms, algorithm), options_solver)
    seed_even = solution_even.minimizer
    solution_uneven = Optim.optimize(ϵ_JOOP_uneven, lc, uc_OOP, seed_even, algorithm), options_solver)

#* Import solutions
    results_triangular_even = BSON.load(datadir("TriangularLattice","2021-03-03_Triangular_lattice_even_AFM.bson"))
    results_triangular_uneven = BSON.load(datadir("TriangularLattice","2021-03-03_Triangular_lattice_uneven_AFM.bson"))
