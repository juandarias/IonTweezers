module pinned_system_ipnewton

using LinearAlgebra

export ϵ_bJ, ϵ_grad!, ϵ_hes!, con_c!, con_jac!, con_hes!, con_c_all!, con_jac_all!, con_hes_all!, glob_vars


"========================="
# Setting global variables
"========================="

function glob_vars(number_ions::Int, number_modes::Int, coupling_modes::Array, target_modes::Array, ev_tmodes::Vector, tmode_indices::Vector, pmodes::Array, ev_pmodes::Vector)
    global Nions = number_ions;
    global Nmodes = number_modes;
    global bJ = coupling_modes;
    global bmprime = target_modes;
    global bm = pmodes;
    global λm = ev_pmodes;
    global λmprime = ev_tmodes;
    global m_index = tmode_indices;
    println("Global variables set!")
end


"===================================="
# Objective function and derivatives
"===================================="

    #### Objective function
    ϵ_bJ(Ω) = sum([sum((bJ[:,J] - bmprime[:,J] - sum([sum([m!=m_index[J] && (bmprime[:,J][i]*Ω[i]*bm[:,m][i])/(λmprime[J] - λm[m]) for i in 1:Nions])*bm[:,m] for m in 1:Nmodes])).^2) for J in 1:Nmodes])

    #### Gradient
    function ϵ_grad!(g, Ω) #where g ϵ ℜ^Nions. It works!
        zero_vector = zeros(Nions);
        Δb(Ω,J) = bJ[:,J] - bmprime[:,J] - sum([sum([m!=m_index[J] && (bmprime[:,J][i]*Ω[i]*bm[:,m][i])/(λmprime[J] - λm[m]) for i in 1:Nions])*bm[:,m] for m in 1:Nmodes]); #if && not enough, follow example of δΔb
        δΔb(i,J) = sum([m!=m_index[J] ? ((bmprime[:,J][i]*bm[:,m][i]/(λmprime[J] - λm[m]))*bm[:,m]) : zero_vector for m in 1:Nmodes]);
        for i in 1:Nions
            g[i] = -sum([sum([2*Δb(Ω,J)[k]*δΔb(i,J)[k] for k in 1:Nions]) for J in 1:Nmodes]) 
        end
    end

    #### Hessian
    function ϵ_hes!(h, Ω) #It also works
        zero_vector = zeros(Nions);
        δΔb(i,J) = sum([m!=m_index[J] ? ((bmprime[:,J][i]*bm[:,m][i]/(λmprime[J] - λm[m]))*bm[:,m]) : zero_vector for m in 1:Nmodes]);
        for i in 1:Nions
            for j in 1:Nions
                h[i,j] = sum([sum([δΔb(i,J)[k]*δΔb(j,J)[k] for k in 1:Nions]) for J in 1:Nmodes])
            end
        end
    end


"======================"
# Constrain functions
"======================"


    ### Only Ω
    con_c!(c, Ω) = (c[1] = sum([Ω[i] for i in 1:Nions]); c)

    function con_jac!(J, Ω) #Oke
        for i in 1:Nions
            J[1,i] = 1;
        end
        J
    end

    function con_hes!(h, Ω, λ) #Oke
        for i in 1:Nions
            h[i,i] += 0;
        end
    end
    
    ### Ω, mu and k1
    
    function con_c_all!(c, Ω) #First constrains for Ω[i] > 0, next for μ, k1 and finally sum(Ω[i]) < Ωmax
        for i in 1:Nions 
            c[i] = Ω[i]
        end
        c[Nions+1] = Ω[Nions+1]
        c[Nions+2] = Ω[Nions+2]
        c[Nions+3] = sum([Ω[i] for i in 1:Nions])
        c
        end

    function con_jac_all!(J, Ω) #Oke
        for i in 1:Nions 
            J[i,i] = 1
        end
        J[Nions+1,Nions+1] = 1
        J[Nions+2,Nions+2] = 1
        for i in 1:Nions
            J[Nions+3,i] = 1;
        end
        J
    end

    function con_hes_all!(h, Ω, λ) #Oke
        for i in 1:Nions+3
            h[i,i] += 0;
        end
    end


end