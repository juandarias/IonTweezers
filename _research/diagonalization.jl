"================"
# Packages and modules
"================"

using QuantumOptics


"================"
# Definitions
"================"

⊗ = kron


"================"
# Hilbert space
"================"


function ManyBodyBasis(number_ions::Int)
    b_mb = []
    #Single particle operator and basis
    bs = SpinBasis(1//2)
    for i in 1:number_ions
        i == 1 ? (b_mb = bs) : (b_mb = b_mb ⊗ bs)
    end
    return b_mb
end


function PauliOperators(basis, ion::Int)
    σx = sigmax(SpinBasis(1//2));
    σy = sigmax(SpinBasis(1//2));
    σz = sigmax(SpinBasis(1//2));
    
    σxi = embed(basis, ion, σx);
    σyi = embed(basis, ion, σy);
    σzi = embed(basis, ion, σz);

    return σxi, σyi, σzi
end


function IsingHamiltonian(coupling_matrix::Array{Float64,2}, basis, number_ions)
    Hx = spzeros(2^number_ions,2^number_ions)
    for i in 1:number_ions
        for j in 1:number_ions
            Hij = (coupling_matrix[i,j]*PauliOperators(basis,i)[1]*PauliOperators(basis,j)[1]).data
            Hx += Hij
        end
    end
    return Hx
end


