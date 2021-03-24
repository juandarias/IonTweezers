module target_modes

push!(LOAD_PATH, "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Code/Julia/2DCrystal/framework/")

using LinearAlgebra, LinearAlgebra.BLAS, SparseArrays, Arpack;
### Own modules
using constants

export TargetModes

function TargetModes(J::Array, phmodes::Array, λ_phmodes::Vector, number_ions::Int)
    bmprime = zeros(number_ions,number_ions)
    λmprime = zeros(number_ions)
    λ_J, ev_J = eigen(J)
    weights = [ev_J[:,i]⋅phmodes[:,j] for j in 1:number_ions,  i in 1:number_ions] # bt = weights * phmodes
    m_index_max = [findmax(abs.(weights[:,i]))[2] for i in 1:number_ions]
    weightmax = [findmax(abs.(weights[:,i]))[1] for i in 1:number_ions]
    for i in 1:number_ions
        bmprime[:,i]= phmodes[:,m_index_max[i]];
        λmprime[i] = λ_phmodes[m_index_max[i]];
    end
    return ev_J, bmprime, λmprime, weightmax, m_index_max
end


function gram_schmidt_new(a; tol = 1e-10) #A is a matrix with eigenvectors as columns
    q = zeros(size(a))
    for i = 1:size(a,2)
        qtilde = a[:,i]
        for j = 1:i-1
            qtilde -= (q[:,j]'*a[:,i]) * q[:,j]
        end
        if norm(qtilde) < tol
            return q
        end
        q[:,i] = qtilde/norm(qtilde)
        #push!(q, qtilde/norm(qtilde))
    end;
    return q
end



end


"============="
# Old methods
"============="

function TargetModesOld(J::Array, phmodes::Array, cutoff::Float64) #Set a cut-off for eigenvalues
    es = eigen(J)
    w2 = [abs(es.values[i]) .< cutoff for i in 1:length(es.values)]
    evaltilde = deleteat!(es.values, w2)
    evectilde = es.vectors[:,.!w2]
    
    return evaltilde, evectilde
    #or use QR decomposition
    # need to extract phonon modes with higher weight
end

