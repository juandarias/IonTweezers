push!(LOAD_PATH, "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Code/Julia/2DCrystal/framework/")
using LinearAlgebra
using constants


"======================"
# Refereces
# 1.James, D. F. V. Quantum dynamics of cold trapped ions, with application to quantum computation. Applied Physics B: Lasers and Optics 66, 181–190 (1998).
"======================"


"============================="
# Hessian of unpinned system
"============================="

ω_trap = [0.2,0.8,0.2]*2*π*10^6 #ωx,ωy,ωz
#r_ions = [[1,2,3], [4,5,6], [7,8,9]] #3 ions coords
r_ions = [[-0.600202, 6.61744*10^-24, -1.28033], [0.808699, 6.61744*10^-24, -1.15996], [-1.4089, 9.92617*10^-24, -0.120376], [-7.85716*10^-17, -3.30872*10^-24, -1.82171*10^-17], [1.4089, 3.30872*10^-24, 0.120376], [-0.808699, 6.61744*10^-24, 1.15996], [0.600202, 6.61744*10^-24, 1.28033]]

l0 = (C_e^2/(4*π*ϵ_0*mYb*ω_trap[3]))^(1/3)

k = ω_trap[3]^2*l0^2*mYb/2
u_ions = r_ions/1


Nions = 7
Hess  = zeros(3*Nions, 3*Nions)

`### Old`
    for m in 1:Nions
        for n in 1:Nions
            if m!=n
                d2V_drndrm[m,n] = -2/(norm(u_ions[m]-u_ions[n])^3) #eq. 3.3 James 1998
                for alpha in 1:3
                    for beta in 1:3
                        Hess[alpha*m, beta*n] = d2V_drndrm[m,n]*u_ions[m][alpha]*u_ions[n]*[beta]/(norm(u_ions[m])*norm(u_ions[n]))
                    end
                end
            else

                dV_dr[3*m-2:3*m] = [sum([k!=m ? (u_ions[m][alpha]-u_ions[k][alpha])/(norm(u_ions[m]-u_ions[k])^3) : 0 for k in 1:Nions]) for alpha in 1:3] #eq. 2.5 James 1998
                #dV_dr[m] =  sum(k!=m ? sign(u_ions[m]-u_ions[k])/(norm(u_ions[m]-u_ions[k])^2) : 0 for k in 1:Nions) #eq. 2.5 James 1998
                #d2Vtilde_dαmdβn = (ω_trap[alpha]/ω_trap[3])^2*(alpha==beta)
                d2V_drndrm[m,m] = 2*sum([p!=m ? 1/(norm(u_ions[m]-u_ions[p])^3) : 0 for p in 1:Nions]) #eq. 3.3 James 1998

                for alpha in 1:3
                    for beta in 1:3
                        Hess[alpha*m, beta*m] = (ω_trap[alpha]/ω_trap[3])^2*(alpha==beta) + d2V_drndrm[m,m]*u_ions[m][alpha]*u_ions[m]*[beta]/(norm(u_ions[m])^2) + dV_dr[m]*((alpha==beta)*norm(u_ions[m])^2-u_ions[m][alpha]*u_ions[m]*[beta])/(norm(u_ions[m])^3)
                    end
                end
            end
        end
    end
"### Old"


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



dVtilde_prime_prime = (ω_trap[1]/ω_trap[3])^2*(1==1);                    
dV_prime_prime = sum([1!=m ? (-(1==1)*norm(u_ions[1]-u_ions[m])^2 + 3*(u_ions[1][1]-u_ions[m][1])*(u_ions[1][1]-u_ions[m][1]))/(norm(u_ions[1]-u_ions[m])^5) : 0 for m in 1:Nions]);
hes11 = dVtilde_prime_prime + dV_prime_prime;


"============================="
# Hessian of Pinned system
"============================="


#Ω2pin_trans_Y(omega) = diagm([mod(i+1,3)==0 ? omega[Int((i+1)/3)] : 0 for i in 1:3*Nions])
#Ω2pin_plane_X(omega) = diagm([mod(i+2,3)==0 ? omega[Int((i+2)/3)] : 0 for i in 1:3*Nions])
#Ω2pin_plane_Z(omega) = diagm([mod(i,3)==0 ? omega[Int((i)/3)] : 0 for i in 1:3*Nions])

Ω2pin_trans_Y(omega) = kron(diagm([0,1,0]), diagm(omega))
Hess_pinned_Y(omega_pin) = Hess + Ω2pin_trans_Y(omega_pin)




"=============================="
# Coupling matrix experimental
"=============================="

λm(omega_pin) = eigen(Hess_pinned_Y(omega_pin)).values
bm(omega_pin) = eigen(Hess_pinned_Y(omega_pin)).vectors
μraman = 2



