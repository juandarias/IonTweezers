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
    
