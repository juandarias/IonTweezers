module symmetries
    module TriangularLattice
    "Auxiliary method that group basis states related by a rotation"

    function SymmetricBasisSetSimple(hadamard_subspace, hadamard_indice)
    
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

    "Generates sets of basis states per hadamard sector related by a rotation, C₆"
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

    "Finds all states in a set (sector) which are equal to a target state but by one single qubit flip"
    function FindNegation(state, sector)
        state = bit_rep(state-1,19)
        hw = 20-sum(state)
        nss = length(sector)

        #* find these pair states in hadamard sector
        pair_states = Vector{Int64}()
        pair_states_indices = Vector{Tuple{Int64,Int64,Int64}}()
        for n=1:19
            pair_state = copy(state)
            pair_state[n] = .!state[n]
            pair_state_number = state_number(pair_state)+1
            for ss=1:nss
                index = findfirst(isequal(pair_state_number),sector[ss])
                if index !== nothing
                    push!(pair_states, sector[ss][index])
                    push!(pair_states_indices, (hw,ss,index))
                end
            end
        end
        return pair_states, pair_states_indices
    end
    end
end
    