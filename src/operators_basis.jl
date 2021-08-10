module operators_basis

    using LinearAlgebra, SparseArrays

    export σᶻᵢσᶻⱼ, σᶻᵢ, σˣᵢ, generate_basis

    #* Operators. Convention bigO = Oₙ ⊗ ... O₂ ⊗ O₁

    function σᶻᵢσᶻⱼ(i,j,N)
        σᶻ = sparse([1 0; 0 -1]);
        II = sparse([1 0; 0 1]);
        i==1 || j==1 ? (op = σᶻ) : (op = II)
        for n=2:N
            n==i || n==j ? (op = kron(σᶻ,op)) : (op = kron(II, op))
        end
        return op
    end

    function σᶻᵢ(i,N)
        σᶻ = sparse([1 0; 0 -1]);
        II = sparse([1 0; 0 1]);
        return kron([n==i ? σᶻ : II for n in N:-1:1]...)
    end

    function σˣᵢ(i,N)
        σˣ = sparse([0 1; 1 0]);
        II = sparse([1 0; 0 1]);
        return kron([n==i ? σˣ : II for n in N:-1:1]...)
    end

    function σʸᵢ(i,N)
        σʸ = sparse([0 -im; im 0]);
        II = sparse([1 0; 0 1]);
        return kron([n==i ? σʸ : II for n in N:-1:1]...)
    end  




    #* Basis
    """
    Binary `BitArray` representation of the given integer `num`, padded to length `N`.
    """
    bit_rep(num::Integer, N::Integer) = Vector{Bool}(digits(num, base=2, pad=N))
    #state_number(state::BitArray) = parse(Int,join(Int64.(state)),base=2)

    """
        generate_basis(N::Integer) -> basis

    Generates a basis (`Vector{BitArray}`) spanning the Hilbert space of `N` spins.
    """
    function generate_basis(N::Integer)
        nstates = 2^N
        basis = Vector{BitArray{1}}(undef, nstates)
        for i in 0:nstates-1
            basis[i+1] = bit_rep(i, N)
        end
        return basis
    end
end


"==="
# Old
"==="


function σᶻᵢσᶻⱼold(i,j,N)
    σᶻ = sparse([1 0; 0 -1]);
    II = sparse([1 0; 0 1]);
    return kron([n==i || n==j ? σᶻ : II for n in 1:N]...)
end