function MagnetizationZ(state) #Wrong. Consider Ψ = |↑↓↑⟩, the result would be M=1
    N = Int(log(length(state))/log(2))
    #M = (1/N)*sum([(state')*σᶻᵢ(i,N)*state for i=1:N])
    M = 0.0
    for i=1:N
        M += abs((state')*σᶻᵢ(i,N)*state) 
    end
    return M/N
end


function TotalMagnetizationZ(state)
    N = Int(log(length(state))/log(2))
    #M = (1/N)*sum([(state')*σᶻᵢ(i,N)*state for i=1:N])
    M = 0.0
    for i=1:N
        M += (state')*σᶻᵢ(i,N)*state
    end
    return M/N
end



function AverageMagnetizationZ(state, basis) #also staggered magnetization, state = ∑cₙ * |basisₙ⟩
    M = 0.
    for (i, bstate) in enumerate(basis)
        bstate_M = 0.
        for spin in bstate
            bstate_M += (state[i]^2 * (spin ? 1 : -1))/length(bstate) #∑ₙ(cₙ²*∑ᵢ⟨basisₙ|σᶻᵢ|basisₙ⟩)
        end
        @assert abs(bstate_M) <= 1
        M += abs(bstate_M) #M = ∑ₙcₙ²*|(∑ᵢ⟨basisₙ|σᶻᵢ|basisₙ⟩)|
    end
    return M
end

function TotalMagnetizationZ(state, basis)
    M = 0.
    for (i, bstate) in enumerate(basis)
        bstate_M = 0.
        for spin in bstate
            bstate_M += (state[i]^2 * (spin ? 1 : -1))/length(bstate) #calculates projection of state in basis state
        end
        @assert abs(bstate_M) <= 1
        M += bstate_M
    end
    return M
end

function ClockOrderParameter(lattice_labels, bstate)
    Oxy = 0.0
    for i in 1:length(bstate)
        lattice_labels[i] == "A" && (Oxy += bstate[i] ? 1 : -1)
        lattice_labels[i] == "B" && (Oxy += bstate[i] ? exp(im*4π/3) : -exp(im*4π/3))
        lattice_labels[i] == "C" && (Oxy += bstate[i] ? exp(-im*4π/3) : -exp(-im*4π/3))
    end
    return [Oxy, abs(Oxy)]
end







"=============="
# Old
"=============="



function MagnetizationZ(state, basis, spin)
    M = 0.
    for (i, bstate) in enumerate(basis)
        M += (state[i]^2 * (bstate[spin] ? -1 : 1)) #calculates projection of state in basis state
    end
    return M
end