using LinearAlgebra


"===================================="
# Objective function and derivatives
"===================================="

#### One traget phonon mode
bt # target mode from diagonalization of J
bk = [i/2 for i in 1:5] #highest weight phonon mode
bl = [[i/2 for i in 1:5] for j in 1:10] #phonon modes
Δbt(Ω) = bt - (bk + [sum([bk[i]*Ω[i]*bl[l][i] for i in 1:5])*bl[l] for l in 1:10]) #l phonon modes, i ions

#### One traget phonon mode
bk = [i/2 for i in 1:5] #highest weight phonon mode
bl = [[i/2 for i in 1:5] for j in 1:10] #phonon modes
Δbt(Ω) = sum(bt[t] - (bk[t] + [sum([bk[t][i]*Ω[i]*bl[l][i] for i in 1:5])*bl[l] for l in 1:10]) for t in 1:tt) #l phonon modes, i ions, tt number of target modes



#### Gradient
function bt_grad!(g)
    g = [sum([bk[i]*bl[l][i] for l in 1:10]) for i in 1:5]
end

#### Hessian
function bt_hessian!(h)
    h = zeros(5,5)
end


"======================"
# Constrain functions
"======================"

con_c!(c, Ω) = (c[1] = sum([Ω[i] for i in 1:10]); c)
function con_jacobian!(J)
    for i in 1:5
        J[1,i] = 1;
    end
    J
end 

function con_h!(h)
    for i in 1:5
        h[i,i] += 0
    end
end;

function con_h!(h)
    h =diagm(zeros(5))
end;
