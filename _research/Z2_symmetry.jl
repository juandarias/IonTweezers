#* Implementing symmetries

    basis = generate_basis(5);

    function Z2Symmetry(basis)
        l = length(basis);
        while l>0
            Z_state = state_number(.!basis[1])
            #deleteat!(basis,1)
            for n =1:N_spins
                pair_state = copy(basis[1])
                pair_state[n] = .!state[1][n]
            end
            l -= 1;
        end
        
    end
