module coupling_matrix

    using LinearAlgebra, LinearAlgebra.BLAS, SparseArrays, Arpack, Calculus, DrWatson;
    include(srcdir()*"/constants.jl")

    export Jtarget, Jexp1D, Jexp3D, JexpAD, JexpSVD, JexpPT, Jexp_multiple_note, Hessian

    function Jtarget(edge_list_FM::Vector, edge_list_AFM::Vector, Nions::Int; J_strength=0.0)
        J = spzeros(Nions,Nions)
        eFM = edge_list_FM
        eAFM = edge_list_AFM

        if J_strength==0.0
            for i in 1:length(edge_list_FM)
                J[eFM[i][1],eFM[i][2]] = -1
            end

            for i in 1:length(edge_list_AFM)
                J[eAFM[i][1],eAFM[i][2]] = 1
            end
        else
            for i in 1:length(edge_list_FM)
                J[eFM[i][1],eFM[i][2]] = -1*J_strength[i]
            end

            for i in 1:length(edge_list_AFM)
                J[eAFM[i][1],eAFM[i][2]] = 1*J_strength[i]
            end
        end

        return J + transpose(J)
    end

    ### Contribution of segmented Paul Trap
    function HessianEquidistantTrap(a::Float64, b::Float64, d::Float64)
        e = ee;
		Vtrap(ux,uy,uz) = a*(d*ux)^2/(4*e) - b*(d*ux)^4/(16*e) - b*(d*uy)^4/(16*e) - b*(d*uz)^4/(4*e) + (d*uy)^2*(2*a - 3*b*(d*ux)^2)/(8*e) + (d*uz)^2*(-2*a + 3*b*(d*ux)^2 + 3*b*(d*uy)^2)/(4*e)
		hVtrap=Calculus.hessian(r -> Vtrap(r[1],r[2],r[3]))
		return hVtrap
    end
    
    ### Hessian unpinned system
    function Hessian(u_ions::Array{Float64}, ω_trap::Array{Float64}; planes::Array{Int64}=[1])
        Nions = size(u_ions)[2];
        length(planes)==3 ? Hess = zeros(3*Nions, 3*Nions) : Hess = zeros(Nions, Nions);
        for α in planes, i in 1:Nions, β in planes, j in 1:Nions
            u = i + (α-1)*Nions*(length(planes)==3); v = j + (β-1)*Nions*(length(planes)==3);
            if i!=j
                r_ij = norm(u_ions[:,i]-u_ions[:,j]);
                ddVharmonic = 0;
                ddVcoulomb = ((α==β)*(r_ij)^2 - 3*(u_ions[α,i]-u_ions[α,j])*(u_ions[β,i]-u_ions[β,j]))/((r_ij)^5);
                Hess[u, v] = ddVharmonic + ddVcoulomb;
            elseif i==j
                ddVharmonic = (ω_trap[α]/ω_trap[3])^2*(α==β);
                ddVcoulomb = sum([i!=k ? (-(α==β)*norm(u_ions[:,i]-u_ions[:,k])^2 + 3*(u_ions[α,i]-u_ions[α,k])*(u_ions[β,i]-u_ions[β,k]))/(norm(u_ions[:,i]-u_ions[:,k])^5) : 0 for k in 1:Nions]);
                Hess[u, v] = ddVharmonic + ddVcoulomb;
            end
        end
        return Hess
    end

    ### Hessian unpinned system homogeneous spacing. Planes: x=1; y=2; z=3
    function Hessian(u_ions::Array{Float64}, ω_trap::Array{Float64}, coeffs::Array{Float64}, size_crystal::Float64; planes::Array{Int64}=[1])
        Nions = size(u_ions)[2]; 
        x0=size_crystal; d=2*x0/Nions; d += d/(Nions -1);
        a,b = coeffs[1], coeffs[2]
        HessTrapEqui = HessianEquidistantTrap(a,b,d)
        length(planes)==3 ? Hess = zeros(3*Nions, 3*Nions) : Hess = zeros(Nions, Nions);
        for α in planes, i in 1:Nions, β in planes, j in 1:Nions
            u = i + (α-1)*Nions*(length(planes)==3); v = j + (β-1)*Nions*(length(planes)==3);
            if i!=j
                r_ij = norm(u_ions[:,i]-u_ions[:,j]);
                ddVharmonic = 0;
                ddVcoulomb = 0.5*((α==β)*(r_ij)^2 - 3*(u_ions[α,i]-u_ions[α,j])*(u_ions[β,i]-u_ions[β,j]))/((r_ij)^5);
                Hess[u, v] = ddVharmonic + ddVcoulomb;
            elseif i==j
                ddvtrapequi = (ee/(d^2*mYb*ω_trap[3]^2))*HessTrapEqui(u_ions[:,i])
                ddVharmonic = (ω_trap[α]/ω_trap[3])^2*(α==β)*(α!=3);
                ddVcoulomb = 0.5*sum([i!=k ? (-(α==β)*norm(u_ions[:,i]-u_ions[:,k])^2 + 3*(u_ions[α,i]-u_ions[α,k])*(u_ions[β,i]-u_ions[β,k]))/(norm(u_ions[:,i]-u_ions[:,k])^5) : 0 for k in 1:Nions]);
                Hess[u, v] = ddVharmonic + ddVcoulomb + ddvtrapequi[α,β];
            end
        end
        return Hess
    end

    ### Definition of AD is according Toffoli paper
    function JexpAD(pos_ion::Array, Ωpin::Vector, ω_trap::Vector, μ_raman::Float64; kvec::Vector=[0,1,0]) #Working!
        Nions = size(pos_ion)[2];
        Nmodes = 3*Nions;
        l0 = (ee^2/(4*π*ϵ0*mYb*ω_trap[3]^2))^(1/3);
        u_ions = pos_ion;
        kvec_ion(numIon) = kron(kvec,[i==numIon ? 1 : 0 for i in 1:Nions]); #raman k-vector on basis of Hessian matrix for target ion

        ### Hessian unpinned system
        Hess = Hessian(u_ions, ω_trap);

        ### Pinning matrix and Hessian of pinned system
        signΩ(omega) = diagm([sign(omega[i]) for i in 1:Nions]);
        Ω2pin_trans_Y(omega) = kron(diagm(kvec), diagm(signΩ(omega)*((omega./ω_trap[3]).^2)));
        Hess_pinned_Y(omega) = Hess + Ω2pin_trans_Y(omega);

        ### Phonon modes and frequencies
        λm, bm = eigen(Hess_pinned_Y(Ωpin)); #combine with evals
        ωm = ω_trap[3]*sqrt.(λm)

        ### Coupling matrix
        Jexp = [sum([i==j ? 0 : (kvec_ion(i) ⋅ bm[:,m])*(kvec_ion(j) ⋅ bm[:,m])/(4*ωm[m]*(μ_raman -ωm[m])) for m in 1:Nmodes]) for i in 1:Nions, j in 1:Nions];#the angular frequency of phonon mode m is ω_m = ω_z*√λ_m, where λ_m is the eigenvalue of mode m.



        return real(Jexp), ωm, bm
    end


    function Jexp1D(pos_ions::Array, Ωpin::Vector, ω_trap::Vector, μ_raman::Float64; equidistant::Bool=false, coeffs_field::Array{Float64}=[1.0,1.0], size_crystal::Float64=1.0, planes::Array{Int64}=[1], hessian::Array{Float64,2}=Array{Float64}(undef, 0, 0)) #Working. Too many calls to eigen!
        Nions = size(pos_ions)[2]
        l0 = (ee^2/(4*π*ϵ0*mYb*ω_trap[3]^2))^(1/3);
        u_ions = pos_ions
        
        ### Hessian unpinned system
        if isempty(hessian)
            equidistant == false && (Hess_nopin = Hessian(u_ions, ω_trap, planes=planes));
            equidistant == true && (Hess_nopin = Hessian(u_ions, ω_trap, coeffs_field, size_crystal, planes=planes));
        else
            Hess_nopin = hessian
        end

        ### Pinning matrix and Hessian of pinned system
        signΩ(omega) = diagm([sign(omega[i]) for i in 1:Nions])
        #Ω2pin_matrix(omega) = kron(diagm(kvec), diagm(signΩ(omega)*((omega./ω_trap[3]).^2)))
        Ω2pin_matrix(omega) = diagm(signΩ(omega)*((omega).^2))
        Hess_pinned(omega) = Hess_nopin + Ω2pin_matrix(omega)

        ### Phonon modes and frequencies
        λm, bm = eigen(Hess_pinned(Ωpin)); #combine with evals
        #ωm = ω_trap[3]*sqrt.(Complex.(λm))

        ### Coupling matrix
        Jexp = [sum([i==j ? 0 : bm[i,m]*bm[j,m]/((μ_raman*ω_trap[3])^2 - ω_trap[3]^2*λm[m]) for m in 1:Nions]) for i in 1:Nions, j in 1:Nions];#the angular frequency of phonon mode m is ω_m = ω_z*√λ_m, where λ_m is the eigenvalue of mode m.

        return real(Jexp), λm, bm
    end

    "Input arrays have dimensions 3xN, frequencies in MHz, and positions are normalized positions"
    function Jexp3D(pos_ion::Array{Float64,2}, Ωpin::Array{Float64,2}, ω_trap::Array{Float64,1}, μ_raman::Float64; equidistant::Bool=false, kvec::Vector=[0,1,0], coeffs_field::Array{Float64}=[1.0,1.0], size_crystal::Float64=1.0) #Working!
        Nions = size(pos_ion)[2]; Nmodes = 3*Nions;
        l0 = (ee^2/(4*π*ϵ0*mYb*ω_trap[3]^2))^(1/3);
        u_ions = pos_ion#/l0;
        Ωpin = Ωpin/ω_trap[3];
        kvec_ion(numIon) = kron(kvec,[i==numIon ? 1 : 0 for i in 1:Nions]); #raman k-vector on basis of Hessian matrix for target ion

        ### Hessian unpinned system
        equidistant == true && (Hess = Hessian(u_ions, ω_trap, planes=[1,2,3]));
        equidistant == false && (Hess = Hessian(u_ions, ω_trap, coeffs_field, size_crystal, planes=[1,2,3]));

        ### Pinning matrix and Hessian of pinned system. The structure of the matrix are sublocks of coordinates, i.e. [XX XY XZ; YX YY YZ; ZX ZY ZZ]
        Ω2pin(omega)=Symmetric(diagm(0=>[(omega[1,:].^2); (omega[2,:].^2); (omega[3,:].^2)],Nions=>[(omega[1,:].*omega[2,:]); (omega[2,:].*omega[3,:])],2*Nions=>(omega[1,:].*omega[3,:])));
        Hess_pinned(omega) = Hess + Ω2pin(omega);


        ### Phonon modes and frequencies
        λm, bm = eigen(Hess_pinned(Ωpin)); #combine with evals
        ωm = ω_trap[3]*sqrt.(Complex.(λm))

        ### Coupling matrix
        Jexp = [sum([i==j ? 0 : (kvec_ion(i) ⋅ bm[:,m])*(kvec_ion(j) ⋅ bm[:,m])/( μ_raman^2 -ωm[m]^2) for m in 1:Nmodes]) for i in 1:Nions, j in 1:Nions];#the angular frequency of phonon mode m is ω_m = ω_z*√λ_m, where λ_m is the eigenvalue of mode m.

        return real(Jexp), ωm, bm
    end


    "Arguments: pos_ion::normalized ion positions; pinning_vector; trap_freqs; beatnote"
    function Jexp_multiple_note(pos_ion::Array, Ωpin::Vector, ω_trap::Vector, μ_raman::Vector; kvec::Vector=[0,1,0]) #Working!
        Nions = length(pos_ion)
        Nmodes = 3*Nions
        l0 = (ee^2/(4*π*ϵ0*mYb*ω_trap[3]^2))^(1/3);
        u_ions = pos_ion
        kvec_ion(numIon) = kron(kvec,[i==numIon ? 1 : 0 for i in 1:Nions]) #raman k-vector on basis of Hessian matrix for target ion

        ### Hessian unpinned system
        Hess  = zeros(Nmodes, Nmodes)
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
        signΩ(omega) = diagm([sign(omega[i]) for i in 1:Nions])
        Ω2pin_trans_Y(omega) = kron(diagm(kvec), diagm(signΩ(omega)*((omega./ω_trap[3]).^2)))
        Hess_pinned_Y(omega) = Hess + Ω2pin_trans_Y(omega)

        ### Phonon modes and frequencies
        ωm(omega) = ω_trap[3]*sqrt.(Complex.(eigen(Hess_pinned_Y(omega)).values)) #PyPlot.scatter(ωm(Ωpin),[1:1:3*Nions;])
        bm(omega) = eigen(Hess_pinned_Y(omega)).vectors

        ### Coupling matrix
        Jexp = zeros(7,7)
        for r in 1:length(μ_raman)
            Jexp += [sum([i==j ? 0 : (kvec_ion(i) ⋅ bm(Ωpin)[:,m])*(kvec_ion(j) ⋅ bm(Ωpin)[:,m])/(μ_raman[r]^2 - ωm(Ωpin)[m]^2) for m in 1:Nmodes]) for i in 1:Nions, j in 1:Nions] #the angular frequency of phonon mode m is ω_m = ω_z*√λ_m, where λ_m is the eigenvalue of mode m.
        end

        return Jexp, ωm(Ωpin), bm(Ωpin)
    end


    function JexpSVD(pos_ion::Array, Ωpin, ω_trap::Vector, μ_raman::Float64;kvec::Vector=[0,1,0]) #ERROR: MethodError: no method matching JexpSVD(::Array{Array{Float64,1},1}, ::Leaf{Array{Float64,1}}, ::Array{Float64,1}, ::Float64)
        Nions = length(pos_ion)
        Nmodes = 3*Nions
        l0 = (ee^2/(4*π*ϵ0*mYb*ω_trap[3]^2))^(1/3);
        u_ions = pos_ion
        kvec_ion(numIon) = kron(kvec,[i==numIon ? 1 : 0 for i in 1:Nions]) #raman k-vector on basis of Hessian matrix for target ion

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
        diag_Ω(Ω) = Matrix(I, Nions, Nions).*(ones(Nions,1)*((Ω/ω_trap[3]).^2)') #ore use kron
        Ω2pin_trans_Y(Ω) = kron(diagm(kvec), diag_Ω(Ω))
        Hess_pinned_Y(Ω) = Hess + Ω2pin_trans_Y(Ω)

        ### Phonon modes and frequencies
        bm(Ω) = svd(Hess_pinned_Y(Ω)).U
        ωm(Ω) = ω_trap[3]*sqrt.(real(svd(Hess_pinned_Y(Ω)).S)) #PyPlot.scatter(ωm(Ωpin),[1:1:3*Nions;])

        ### Coupling matrix
        Jexp = [sum([i==j ? 0 : (kvec_ion(i) ⋅ bm(Ωpin)[:,m])*(kvec_ion(j) ⋅ bm(Ωpin)[:,m])/( μ_raman^2 -ωm(Ωpin)[m]^2) for m in 1:Nmodes]) for i in 1:Nions, j in 1:Nions] #the angular frequency of phonon mode m is ω_m = ω_z*√λ_m, where λ_m is the eigenvalue of mode m.

        return Jexp
    end


    function JexpPT(pos_ion::Array, Ωpin::Vector, ω_trap::Vector, μ_raman::Float64; kvec::Vector=[0,1,0])
        Nions = length(pos_ion)
        Nmodes = 3*Nions
        l0 = (ee^2/(4*π*ϵ0*mYb*ω_trap[3]^2))^(1/3);
        u_ions = pos_ion#/l0
        kvec_ion(numIon) = kron(kvec,[i==numIon ? 1 : 0 for i in 1:Nions]) #raman k-vector on basis of Hessian matrix for target ion

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

        ### Phonon modes and values unpinned system
        λm = eigen(Hess).values #PyPlot.scatter(ωm(Ωpin),[1:1:3*Nions;])
        bm = eigen(Hess).vectors

        ### Modified phonon modes and values pinned system
        bn_tilde(Ω,n) = bm[:,n] + sum([sum([m!=n && (bm[:,n][i]*(Ω[i])^2*bm[:,m][i])/(ωm[n] - ωm[m]) for i in 1:Nions])*bm[:,m] for m in 1:Nmodes])

        λn_tilde(Ω,n) = λm[n] + sum([(bm[:,n][i]*(Ω[i])^2*bm[:,n][i]) for i in 1:Nions]) + sum([sum([m!=n && (bm[:,n][i]*(Ω[i])^2*bm[:,m][i])^2/(λm[n] - λm[m]) for i in 1:Nions]) for m in 1:Nmodes])

        ωn(Ω,n) = ω_trap[3]*sqrt(λn_tilde(Ω,n))

        ### Coupling matrix
        Jexp = [sum([i==j ? 0 : (kvec_ion(i) ⋅ bn_tilde(Ωpin,n))*(kvec_ion(j) ⋅ bn_tilde(Ωpin,n))/(μ_raman^2 - ωn(Ωpin,n)^2) for n in 1:Nmodes]) for i in 1:Nions, j in 1:Nions] #the angular frequency of phonon mode m is ω_m = ω_z*√λ_m, where λ_m is the eigenvalue of mode m.

        return Jexp
    end

end



"====="
# Old #
"====="

function Jexp1DOld(pos_ion::Array, Ωpin::Vector, ω_trap::Vector, μ_raman::Float64; kvec::Vector=[0,1,0]) #Working. Too many calls to eigen!
    Nions = length(pos_ion)
    Nmodes = 3*Nions
    l0 = (ee^2/(4*π*ϵ0*mYb*ω_trap[3]^2))^(1/3);
    u_ions = pos_ion
    kvec_ion(numIon) = kron(kvec,[i==numIon ? 1 : 0 for i in 1:Nions]) #raman k-vector on basis of Hessian matrix for target ion

    ### Hessian unpinned system
    equidistant == True && (Hess = Hessian(u_ions, ω_trap));
    equidistant == False && (Hess = Hessian(u_ions, ω_trap, coeffs_field, z0));

    ### Pinning matrix and Hessian of pinned system
    signΩ(omega) = diagm([sign(omega[i]) for i in 1:Nions])
    Ω2pin_trans_Y(omega) = kron(diagm(kvec), diagm(signΩ(omega)*((omega./ω_trap[3]).^2)))
    Hess_pinned_Y(omega) = Hess + Ω2pin_trans_Y(omega)

    ### Phonon modes and frequencies
    ωm(omega) = ω_trap[3]*sqrt.(Complex.(eigen(Hess_pinned_Y(omega)).values)) #PyPlot.scatter(ωm(Ωpin),[1:1:3*Nions;])
    bm(omega) = eigen(Hess_pinned_Y(omega)).vectors #combine with evals

    ### Coupling matrix
    Jexp = [sum([i==j ? 0 : (kvec_ion(i) ⋅ bm(Ωpin)[:,m])*(kvec_ion(j) ⋅ bm(Ωpin)[:,m])/( μ_raman^2 -ωm(Ωpin)[m]^2) for m in 1:Nmodes]) for i in 1:Nions, j in 1:Nions] #the angular frequency of phonon mode m is ω_m = ω_z*√λ_m, where λ_m is the eigenvalue of mode m.

    return real(Jexp), ωm(Ωpin), bm(Ωpin)
end
