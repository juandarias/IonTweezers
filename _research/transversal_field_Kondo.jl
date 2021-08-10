# TODO #############################################################################
# -implement LinearMaps, see: https://www.tensors.net/exact-diagonalization
# -build Hamiltonian for low energy sector of TFIM

#* Load modules
    using DrWatson, BSON, LinearAlgebra, Arpack, DelimitedFiles, PyCall, PyCallUtils, SparseArrays, Revise
    push!(LOAD_PATH, srcdir());
    push!(LOAD_PATH, scriptsdir());
    include(srcdir()*"/hamiltonians.jl");
    include(srcdir()*"/observables.jl");
    using operators_basis

#* Import solutions

    JAFM_19 = readdlm(datadir("TriangularLattice", "AFM_Triangular_Coupling_Matrix.dat"));

#* Build Hamiltonian

    field = 0.00*ones(19);
    HIsing = TransversalIsing(JAFM_19, field)

#* Diagonalize Arpack.
    
    #* Compare eigs vs eigsh
        #* results do not agree. Stick to eigs as no time advantage is present in eigsh
    maxiter = 500;
    λ0s, ϕ0s = eigs(Symmetric(HIsing), nev=1, which=:SR, maxiter = maxiter)#, tol=1E-5)
    λ0, ϕ0 = eigs(HIsing, nev=1, which=:SR, maxiter = maxiter)#, tol=1E-5)


    #* Convergence
    ϵ0 = eps(real(eltype(HIsing)))/2
    χ = norm(HIsing*ϕ0 - λ0[1]*ϕ0)/norm(ϕ0)

    println( "Convergence criterion test for smallest eigenvalue: ", χ <= ϵ0 * max(cbrt(ϵ0^2),abs(λ0[1]))) #! results in false for both cases

#* Compare with Python implemenation of Arpack

    np = pyimport("numpy")
    linalg = pyimport("scipy.sparse.linalg")
    
    for n=1:3
        λ0, ϕ0 = eigs(Hermitian(HIsing), nev=1, which=:SR, v0=ones(2^19))
        λ0, ϕ0r = eigs(Hermitian(HIsing), nev=1, which=:SR, v0=ones(2^19))
        #λ0p, ϕ0p = linalg.eigs(PyObject(HIsing),1,which=:SR) 
        #λ0p, ϕ0pr = linalg.eigs(PyObject(HIsing),1,which=:SR) 
        println(ϕ0r ⋅ ϕ0) #! the ground state overlap varies per run
        #println(ϕ0pr ⋅ ϕ0p) #! the ground state overlap varies per run
    end


    #* Convergence
    ϵ0 = eps(real(eltype(HIsing)))/2
    χ = norm(HIsing*ϕ0p - λ0p[1]*ϕ0p)/norm(ϕ0p)

    println( "Convergence criterion test for smallest eigenvalue: ", χ <= ϵ0 * max(cbrt(ϵ0^2),abs(λ0p[1]))) #! results in false

#* Tuning solver parameters
    #* -shift inverse method not needed, as ground state has a large negative energy
    #* -max iters:5000 -> converged
    #* -max iters:2000 -> not converged
    #* -max iters:1500 -> not converged
    #* -max iters:1000 -> not converged
    #* -max iters:500 -> not converged
    #! 25/03: convergence dropped down after doing magnetization calculations of next section. Not even increasing the number of iteration corrected this issue. As if something in memory was affecting the diagonalization. Strange!!!
    #! 26/03: by fixing the starting vector for the Lanczos iterations to ones(N), the problems of convergence seem to have been fixed

    iterations = 500
    for n=1:3
        λ0, ϕ01 = eigs(HIsingF, nev=1, which=:SR, maxiter=iterations)
        #λ0, ϕ02 = eigs(HIsingF, nev=1, which=:SR, maxiter=iterations)
        λ0p, ϕ0p1 = linalg.eigs(PyObject(HIsingF),1,which=:SR, maxiter=iterations) 
        #λ0p, ϕ0p2 = linalg.eigs(PyObject(HIsingF),1,which=:SR, maxiter=iterations) 
        println(ϕ0p1[:,1] ⋅ ϕ01[:,1]) #! confirm convergence
        #println(ϕ0p1 ⋅ ϕ0p2) #! confirm convergence
    end


#* Native Julia diagonalization

    
    using ArnoldiMethod

    function eigen_sparse(x, num_states)
        decomp, history = partialschur(x, nev=num_states, which=SR()); # only solve for the ground state
        vals, vecs = partialeigen(decomp);
        return vals, vecs
    end

    λAM0, ϕAM0 = eigen_sparse(HIsing, 1)
    λAM, ϕAM = eigen_sparse(HIsing, 19)



    λTI = zeros(8,11)
    ϕTI = zeros(2^19,8,11)
    for h=0:10
        h0 = h/10;
        field = h0*ones(19);
        HIsing = TransversalIsing(JAFM_19, field)
        λ, ϕ = eigs(HIsing, nev=8, which=:SR)#, tol=1E-5)
        λTI[:,h+1] = λ
        ϕTI[:,:,h+1] = ϕ
    end

#* Observables: Magnetization

    basis = generate_basis(19);
    Mz0 = MagnetizationZ(ϕ0[:,1])
    Mz0_2 = MagnetizationZ(ϕ0, basis) #! more efficient

    Mz0_vs_h = zeros(20);
    TMz0_vs_h = zeros(10);
    #TMz0_vs_h_o = zeros(10);
    h = 10 .^(range(-2.0, 0, length=20));
    for i=1:20
        overlap = 0.0;
        field = h[i]*ones(19);
        HIsing = TransversalIsing(JAFM_19, field);
        #while overlap < 0.9
            λ0, ϕ0 = eigs(HIsing, nev=1, which=:SR, v0=ones(2^19));
        #    λ0, ϕ0r = eigs(HIsing, nev=1, which=:SR, v0=ones(2^19));
            groundstate = @view ϕ0[:,1];
        #    overlap = abs(ϕ0r[:,1] ⋅ ϕ0[:,1]);
        #    println(overlap)
        #end
        Mz0_vs_h[i] = MagnetizationZ(groundstate, basis)
        #TMz0_vs_h[i] = TotalMagnetizationZ(groundstate, basis)
        #TMz0_vs_h_o[i] = TotalMagnetizationZ(groundstate)
        println(Mz0_vs_h[i])
    end

    results_uneven = BSON.load(datadir("TriangularLattice","2021-03-29_19_ions_uneven_field_scan.bson"))
    results_target = BSON.load(datadir("TriangularLattice","2021-03-29_19_ions_target_field_scan.bson"))
    
    mz_target = results_target[:magnetization]
    mz_uneven = results_uneven[:magnetization]
    
    ht = 10 .^(range(-2.0, 0, length=10));
    fig, ax = plt.subplots()
    scatter(h,mz_uneven,label="exp");
    scatter(ht,mz_target,label="target");
    xscale("log");xlabel("\$b/J\$");ylabel("\$M_z\$");legend()
    savefig(plotsdir("TFIM", string(Dates.today())*"_19_ions_comp_Mz.png"));gcf()

    

#* Observables: Correlators
    #* Nearest neighbor

    using models.Kondo, models.general

    d₀ = 1.5;
    depth = 2;
    pos_ions_even = TriangularLattice(d₀);
    NN, iNN = NearestNeighbour(pos_ions_even, 1.6);

    NNcorr_vs_h = zeros(19,19,10);
    #Mz0_vs_h = zeros(10);
    h = 10 .^(range(-2.0, 0, length=10));
    for i=1:10
        field = h[i]*ones(19);
        HIsing = TransversalIsing(JAFM_19, field);
        λ0, ϕ0 = eigs(HIsing, nev=1, which=:SR, v0=ones(2^19));
        groundstate = @view ϕ0[:,1];
        NNcorr_vs_h[:,:,i] = NeighbourCorrelatorZ(groundstate, NN)
        #Mz0_vs_h[i] = MagnetizationZ(groundstate, basis)
    end

    results_triangular_uneven = BSON.load(datadir("TriangularLattice","2021-03-03_Triangular_lattice_uneven_AFM.bson"))
    position_ions = results_triangular_uneven[:position_ions]

    
    for n=1:10
        f = round(log10(h[n]),digits=2)
        GraphCoupling(NNcorr_vs_h[:,:,n], position_ions, plane="YZ", zero_offset=0.05, label="\$\\langle \\hat\\sigma_i^z \\hat\\sigma_j^z\\rangle\$", title_plot="\$h=10^{$f}\$");
        savefig(plotsdir("TFIM", string(Dates.today())*"_19ions_NNcorr_"*string(n)*".png"))
    end

#* Spectra
    #* Uneven
    energy_vs_h = zeros(6,20);
    states_vs_h = zeros(2^19,6,20);
    h = 10 .^(range(-2.0, 0, length=20));
    for i=1:20
        field = h[i]*ones(19);
        HIsing = TransversalIsing(JAFM_19, field);
        λ, ϕ = eigs(HIsing, nev=6, which=:SR, v0=ones(2^19));
        energy_vs_h[:,i] = λ
        states_vs_h[:,:,i] = ϕ
    end

    #* Target
    energy_vs_h_tar = zeros(6,20);
    states_vs_h_tar = zeros(2^19,6,20);
    h = 10 .^(range(-2.0, 0, length=20));
    for i=1:20
        field = h[i]*ones(19);
        HIsing = TransversalIsing(Float64.(NN), field);
        λ, ϕ = eigs(HIsing, nev=6, which=:SR, v0=ones(2^19));
        energy_vs_h_tar[:,i] = λ
        states_vs_h_tar[:,:,i] = ϕ
    end

    function PlotEnergyGap(energies_experimental, energies_target, parameter)
        nev = size(energies_experimental)[1]
        fig, ax = plt.subplots()
        ax.set_ylabel("\$E/J\$");
        ax.set_xlabel("\$h/J\$");
        ax.set_xscale("log")
        Rescale(energies) = (energies .- maximum(abs.(energies)))/maximum(abs.(energies))
        
        markers = ["x","+","v","^","<",">"]
        for n=1:nev
            ax.plot(parameter, energies_experimental[n,:], marker=markers[n], markersize=7, lw=0.5, fillstyle="none", label="\$E_{\\textrm{exp}}^$(n-1)\$")
            println(n)
        end
        
        PyPlot.gca().set_prop_cycle(nothing)
        
        for n=1:nev
            ax.plot(parameter, energies_target[n,:], marker=markers[n], markersize=7, lw=0.5, fillstyle="none", ls="--")#,label="\$E_{\\textrm{T}}^$(n-1)\$")
        end
        ax.legend()
    end

    PlotEnergyGap(energy_vs_h, energy_vs_h_tar,h);gcf()
    
    savefig(plotsdir("TFIM", string(Dates.today())*"_19_ions_low_energy_spectra_.svg"))

    NormalPlot()

#* Implementing symmetries


    #* Group basis states by Hadamard gate
    hadamard_subspaces = [];
    hadamard_indices = [];
    hw = map(x->sum(x),basis)
    for w=0:19
        hw_indices = findall(hw .== w)
        push!(hadamard_subspaces, basis[hw_indices])
        push!(hadamard_indices, hw_indices)
    end
   
        #Crystal2D(pos_ions_even, Float64.(R6*hadamard_subspaces[2][1]), plane="YZ");gcf()
    #Crystal2D(pos_ions_even, Float64.(hadamard_subspaces[2][3]), plane="YZ");gcf()

    
    @everywhere function SymmetricBasisSetFast(hadamard_subspace, hadamard_indice)
        R6 = zeros(19,19)
        for n=1:19
            n==1 && (R6[1,1] = 1)
            if 2 <= n <= 7
                R6[n, mod(n-2+1,6)+2] = 1
            elseif 8 <= n <= 19
                R6[n, mod(n-8+2,12)+8] = 1
            end
        end
        collection_subspaces = Array{Int64,1}[]
        l = length(hadamard_subspace)
        while l >= 1
            subblock = Int64[];
            sr = Set([(R6^n)*hadamard_subspace[1] for n=1:5]); #* rotate basis states
            push!(subblock,hadamard_indice[1]) #* add label to corresponding conserved subspace
            deleteat!(hadamard_subspace,1) #* remove state from subgroup
            deleteat!(hadamard_indice,1) #* remove label state from label collection
            
            indices_degenerates = hadamard_subspace .∈ Ref(sr)
            push!(subblock, hadamard_indice[indices_degenerates]...)
            push!(collection_subspaces, subblock)
            
            deleteat!(hadamard_subspace, indices_degenerates);
            deleteat!(hadamard_indice, indices_degenerates);
            l -= sum(indices_degenerates) +1
        end
        return collection_subspaces
    end

    @everywhere function SymmetricBasisSetSimple(hadamard_subspace, hadamard_indice)
        
        ### Create rotation matrix
        R6 = zeros(19,19)
        for n=1:19
            n==1 && (R6[1,1] = 1)
            if 2 <= n <= 7
                R6[n, mod(n-2+1,6)+2] = 1
            elseif 8 <= n <= 19
                R6[n, mod(n-8+2,12)+8] = 1
            end
        end

        ### Find symmetric states in each Hadamard sector
        #collection_subspaces = Array{Int64,1}[]
        collection_subspaces = Vector{Array{Int64,1}}()
        l = length(hadamard_subspace)
        while l >= 1
            subblock = Int64[];
            sr = [(R6^n)*hadamard_subspace[1] for n=1:5]; #* rotate basis states
            push!(subblock,hadamard_indice[1]); #* add label to corresponding conserved subspace
            deleteat!(hadamard_subspace,1); #* remove state from subgroup
            deleteat!(hadamard_indice,1); #* remove label state from label collection
            l -= 1; #* reduced length subgroup
            m = 1; #* initiate counter on rremaining states
            while m <= l
                if hadamard_subspace[m] ∈ sr
                    push!(subblock, hadamard_indice[m]); # I am missing a delete statement
                    deleteat!(hadamard_subspace, m);
                    deleteat!(hadamard_indice, m);
                    l -= 1;
                else
                    m += 1;
                end
            end
            push!(collection_subspaces, subblock)
        end
        return collection_subspaces
    end

    function SymmetricBasisParallel(basis)
        #* Group basis states by Hadamard gate
        hadamard_subspaces = [];
        hadamard_indices = [];
        hw = map(x->sum(x),basis)
        for w=0:19
            hw_indices = findall(hw .== w)
            push!(hadamard_subspaces, basis[hw_indices])
            push!(hadamard_indices, hw_indices)
        end
        
        sorted_basis =  @distributed (append!) for w=1:20
            [SymmetricBasisSetSimple(hadamard_subspaces[w], hadamard_indices[w])]
        end
        return sorted_basis
    end

    sym_basis = SymmetricBasisParallel(basis)
        
    coupled_states, indices_coupled_states = FindNegation(sym_basis[3][1][1],sym_basis[2])

    function ReducedBasis(symmetric_basis)
        reduced_basis = Vector{Tuple{Int64,Int64}}();
        reduced_basis_sector = [];
        nw = length(symmetric_basis);
        #Sweep forward
        for w=1:nw
            nss = length(symmetric_basis[w])
            for ss=1:nss
                pair_states, indices_pair = FindNegation(symmetric_basis[w][ss][1], symmetric_basis[nw-w+1]) 
                push!(reduced_basis, (symmetric_basis[w][ss][1],pair_state))
                push!(reduced_basis_sector, ((w,ss,1),index_pair))
            end
        end
        if isodd(nw)
            nss = length(symmetric_basis[nw÷2+1])
            for ss=1:nss
                pair_state, index_pair = FindNegation(symmetric_basis[nw÷2+1][ss][1], symmetric_basis[nw÷2+1]) 
                push!(reduced_basis, (symmetric_basis[nw÷2+1][ss][1],pair_state))
                push!(reduced_basis_sector, ((nw÷2+1,ss,1),index_pair))
            end
        end
        return reduced_basis, reduced_basis_sector
    end


#* Build Hamiltonian symmetric sector

    function TransversalIsingHamiltonian(coupling_matrix, basis_pairs)
        nspins = 19
        dims = 2*length(basis_pairs)
        Hamiltonian = spzeros(dims,dims)
        for n in 1:2:dims ÷ 2
            state_1 = bit_rep(basis_pairs[n][1]-1, nspins)
            state_2 = bit_rep(basis_pairs[n][2]-1, nspins)
            Hamiltonian[2n-1,2n-1] = sum([coupling_matrix[i,j]*(state_1[i] ? 1 : -1)*(state_1[j] ? 1 : -1) for i=1:nspins, j=1:nspins])
            Hamiltonian[2n,2n] = sum([coupling_matrix[i,j]*(state_2[i] ? 1 : -1)*(state_2[j] ? 1 : -1) for i=1:nspins, j=1:nspins])
        end
        return Hamiltonian
    end










##############################################################
# Old attempt of using symmetries to reduce Hilbert Space
# -the function UnitCell seems to be useful
# -the rest is wrong, as ignoring the spins outside the unitcell
# results in a system not equivalent to the original one
##############################################################





    ###########################################
    # Old
    ###########################################

    

    function SymmetricBasisSetFast(hadamard_subspaces, hadamard_indices)
        sorted_basis_set = []
        ns = length(hadamard_subspaces)
        for w=1:ns
            collection_subspaces = Array{Int64,1}[]
            l = length(hadamard_subspaces[w])
            while l >= 1
                subblock = Int64[];
                sr = Set([(R6^n)*hadamard_subspaces[w][1] for n=1:5]); #* rotate basis states
                push!(subblock,hadamard_indices[w][1]) #* add label to corresponding conserved subspace
                deleteat!(hadamard_subspaces[w],1) #* remove state from subgroup
                deleteat!(hadamard_indices[w],1) #* remove label state from label collection
                
                indices_degenerates = hadamard_subspaces[w] .∈ Ref(sr)
                push!(subblock, hadamard_indices[w][indices_degenerates]...)
                push!(collection_subspaces, subblock)
                
                deleteat!(hadamard_subspaces[w], indices_degenerates);
                deleteat!(hadamard_indices[w], indices_degenerates);
                l -= sum(indices_degenerates) +1
            end
            push!(sorted_basis_set, collection_subspaces)
        end
        return sorted_basis_set
    end

    function SymmetricBasisSet(hadamard_subspaces, hadamard_indices)
        R6 = zeros(19,19)
        for n=1:19
            n==1 && (R6[1,1] = 1)
            if 2 <= n <= 7
                R6[n, mod(n-2+1,6)+2] = 1
            elseif 8 <= n <= 19
                R6[n, mod(n-8+2,12)+8] = 1
            end
        end
        sorted_basis_set = []
        ns = length(hadamard_subspaces)
        for w=1:ns
            collection_subspaces = Array{Int64,1}[]
            l = length(hadamard_subspaces[w])
            while l >= 1
                subblock = Int64[];
                sr = [(R6^n)*hadamard_subspaces[w][1] for n=1:5]; #* rotate basis states
                push!(subblock,hadamard_indices[w][1]); #* add label to corresponding conserved subspace
                deleteat!(hadamard_subspaces[w],1); #* remove state from subgroup
                deleteat!(hadamard_indices[w],1); #* remove label state from label collection
                l -= 1; #* reduced length subgroup
                m = 1; #* initiate counter on rremaining states
                while m <= l
                    if hadamard_subspaces[w][m] ∈ sr
                        push!(subblock, hadamard_indices[w][m]); # I am missing a delete statement
                        deleteat!(hadamard_subspaces[w], m);
                        deleteat!(hadamard_indices[w], m);
                        l -= 1;
                    else
                        m += 1;
                    end
                end
                push!(collection_subspaces, subblock)
            end
            push!(sorted_basis_set, collection_subspaces)
        end
        return sorted_basis_set
    end
    
    resfast = SymmetricBasisSetFast(hadamard_subspaces, hadamard_indices)
    res = SymmetricBasisSet(hadamard_subspaces, hadamard_indices) #fastest
    
    
    

    
    
    
