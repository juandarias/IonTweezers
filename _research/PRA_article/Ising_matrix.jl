using LinearAlgebra


⊗ = kron
∑ = sum
diagless(matrix) = matrix - diagm(diag(matrix))

function σᶻᵢⱼ(i,j,N)
    σᶻ = [1 0; 0 -1];
    II = [1 0; 0 1];
    return kron([n==i || n==j ? σᶻ : II for n in 1:N]...)
end

function σᶻᵢ(i,N)
    σᶻ = [1 0; 0 -1];
    II = [1 0; 0 1];
    return kron([n==i ? σᶻ : II for n in 1:N]...)
end

function σˣᵢ(i,N)
    σˣ = [0 1; 1 0];
    II = [1 0; 0 1];
    return kron([n==i ? σˣ : II for n in 1:N]...)
end


Jij = rand(4,4)
Jij = Jij + Jij'
J2 = diagless(Jij)
h = rand(4)

IsingOperator = ∑([J2[i,j]*σᶻᵢⱼ(i,j,4) for i=1:4 for j=1:4])
IsingTransversal = ∑([J[i,j]*σᶻᵢⱼ(i,j,4) for i=1:4 for j=1:4]) + ∑([h[i]*σˣᵢ(i,4) for i=1:4])

diag(IsingOperator)

evals, evecs = eigen(IsingOperator)
evalsT, evecsT = eigen(IsingTransversal)

[(evecs[:,1]')*σᶻᵢ(i,4)*evecs[:,1] for i=1:4]
[(evecs2[:,1]')*σᶻᵢ(i,4)*evecs2[:,1] for i=1:4]

IsingOperator[1:6,1:6]





