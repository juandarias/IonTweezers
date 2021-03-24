"====================================" 
# TODO
# - pass eigenmodes of unpinned system as argument to objective function
# - test effect of different structure of arguments of objective function on optimizer
"====================================" 

module pinned_system_nlopt

using LinearAlgebra, Nabla;

export ϵ_bJ, Ω_cons, ϵ_JSVD;


"========================================"
# Objective function and derivatives: PMO
"========================================"

    function ϵ_bJ(Ω::Vector,g::Vector)
        if length(g) > 0
            zero_vector = zeros(7);
            Δb(Ω,J) = bJ[:,J] - bmprime[:,J] - sum([sum([m!=m_index[J] && (bmprime[:,J][i]*Ω[i]*bm[:,m][i])/(λmprime[J] - λm[m]) for i in 1:Nions])*bm[:,m] for m in 1:Nmodes]); #if && not enough, follow example of δΔb
            δΔb(i,J) = sum([m!=m_index[J] ? ((bmprime[:,J][i]*bm[:,m][i]/(λmprime[J] - λm[m]))*bm[:,m]) : zero_vector for m in 1:Nmodes]);
            for i in 1:Nions
                g[i] = -sum([sum([2*Δb(Ω,J)[k]*δΔb(i,J)[k] for k in 1:Nions]) for J in 1:Nmodes]) 
            end
        end
        return sum([sum((bj[:,J] - bmprime[:,J] - sum([sum([m!=m_index[J] && (bmprime[:,J][i]*Ω[i]*bm[:,m][i])/(λmprime[J] - λm[m]) for i in 1:Nions])*bm[:,m] for m in 1:Nmodes])).^2) for J in 1:Nmodes])
    end


"========================================="
# Objective function and derivatives: CMO
"========================================="

    function ϵ_JSVD(Ω::Vector,g::Vector,Jtarget::Array,Jexp::Function) 
        Nions = length(Ω)
        ### Objective function
        #ϵ_J(Ω) = norm(Jexp(Ω)-Jtarget) #removed as nabla fails on norm operator
        ϵ_J(Ω) = sqrt(sum((Jexp(Ω)-Jtarget).^2))
        if length(g) > 0
            for i in 1:Nions
                g[i] = Nabla.∇(ϵ_J)(Ω)[1][i] 
            end
        end
        return ϵ_J(Ω)
    end


    function ϵ_JSVD2(Ω::Vector,g::Vector,Jtarget::Array) #requires clean up and testing
        Nions = length(pos_ion)
        l0 = (C_e^2/(4*π*ϵ_0*mYb*ω_trap[3]))^(1/3)
        u_ions = pos_ion/l0
        μ_prime = μ_raman/(ω_trap[3]*l0)
    
        ### Hessian unpinned system
        Hess  = zeros(3*Nions, 3*Nions)
        for i in 1:Nions
            for j in 1:Nions
                if i!=j
                    r_ij = norm(u_ions[i]-u_ions[j])
                    for alpha in 1:3
                        for beta in 1:3
                            u = i + (alpha-1)*Nions; v = j + (beta-1)*Nions;
                            dVtilde_prime_prime = 0;
                            dV_prime_prime = ((alpha==beta)*(r_ij)^2 - 3*(u_ions[i][alpha]-u_ions[j][alpha])*(u_ions[i][beta]-u_ions[j][beta]))/((r_ij)^5);
                            Hess[u, v] = dVtilde_prime_prime + dV_prime_prime;
                        end
                    end
                elseif i==j
                    for alpha in 1:3
                        for beta in 1:3
                            u = i + (alpha-1)*Nions; v = j + (beta-1)*Nions;
                            dVtilde_prime_prime = (ω_trap[alpha]/ω_trap[3])^2*(alpha==beta);
                            dV_prime_prime = sum([i!=m ? (-(alpha==beta)*norm(u_ions[i]-u_ions[m])^2 + 3*(u_ions[i][alpha]-u_ions[m][alpha])*(u_ions[i][beta]-u_ions[m][beta]))/(norm(u_ions[i]-u_ions[m])^5) : 0 for m in 1:Nions]);
                            Hess[u, v] = dVtilde_prime_prime + dV_prime_prime;
                        end
                    end
                end
            end
        end
        
        ### Pinning matrix and Hessian of pinned system
        kvec = [0,1,0] # raman k-vector, i.e. [0,1,0] indicates coupling with y-modes
        Ω2pin_trans_Y(omega) = kron(diagm(kvec), diagm(omega))
        Hess_pinned_Y(omega) = Hess + Ω2pin_trans_Y(omega)
    
        ### Phonon modes and frequencies
        λm(omega), bm(omega) = svd(Hess_pinned_Y(omega)).S, svd(Hess_pinned_Y(omega)).U #PyPlot.scatter(λm(omega_pin),[1:1:3*Nions;])        
    
        ### Coupling matrix
        kvec_ion(numIon) = kron(kvec,[i==numIon ? 1 : 0 for i in 1:Nions]) #raman k-vector on basis of Hessian matrix for target ion
        
        JexpSVG = [sum([i==j ? 0 : (kvec_ion(i) ⋅ bm(omega_pin)[:,m])*(kvec_ion(j) ⋅ bm(omega_pin)[:,m])/(λm(omega_pin)[m] - μ_prime^2) for m in 1:3*Nions]) for i in 1:Nions, j in 1:Nions] #the angular frequency of phonon mode m is ω_m = ω_z*l0*√λ_m, where λ_m is the eigenvalue of mode m.
        
        ### Objective function
        JhexSVG(Ω) = JexpSVG(pos_ions, Ω, ω_trap, μ_raman)
        ϵ_JSVG(Ω) = norm(JhexSVG(Ω)-Jtarget)

        if length(g) > 0
            for i in 1:Nions
                g[i] = ∇(ϵ_JSVG)[1][i] 
            end
        end
        return ϵ_JSVG(Ω)
    end

"==============================================================================="
# Constrain functions: lower bounds and upper bounds of each parameter are set with upper_bounds!(lower_bounds! methods)
"==============================================================================="


    function Ω_cons(Ω::Vector,grad::Vector,Ωmax::Float64) #Test if omegamax can be passed as an optional argument
        if length(grad) > 0
            for i in 1:Nions
                grad[i] = 1
            end
        end
        return sum([Ω[i] for i in 1:Nions]) - Ωmax # equivalent to sum([Ω[i] for i in 1:10]) < Ωmax
    end

end
    
    


