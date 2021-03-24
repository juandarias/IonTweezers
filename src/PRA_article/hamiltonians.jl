module hamiltonians
    
    using LinearAlgebra, SparseArrays

    function σᶻᵢⱼ(i,j,N)
        σᶻ = sparse([1 0; 0 -1]);
        II = sparse([1 0; 0 1]);
        return kron([n==i || n==j ? σᶻ : II for n in 1:N]...)
    end

    function σᶻᵢ(i,N)
        σᶻ = sparse([1 0; 0 -1]);
        II = sparse([1 0; 0 1]);
        return kron([n==i ? σᶻ : II for n in 1:N]...)
    end

    function σˣᵢ(i,N)
        σˣ = sparse([0 1; 1 0]);
        II = sparse([1 0; 0 1]);
        return kron([n==i ? σˣ : II for n in 1:N]...)
    end

    function TransversalIsing(coupling_matrix, field)
        N = size(coupling_matrix)[2]
        return sum([coupling_matrix[i,j]*σᶻᵢⱼ(i,j,N) for i=1:N for j=1:N]) + sum([field[i]*σˣᵢ(i,N) for i=1:N])
    end

end