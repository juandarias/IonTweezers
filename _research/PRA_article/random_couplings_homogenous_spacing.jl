using DrWatson, PyPlot, LinearAlgebra, JuMP
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using optimizers, coupling_matrix, crystal_geometry, plotting_functions



"========================="
# Auxiliary functions
"========================="

∑ = sum
⊗(A,B) = kron(A',B)
Chop_Sort(x) = sort(round.(x,digits=5), dims=2)
Diagless(A) = A - diagm(diag(A))
DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./maximum(coupling_matrix)

"===================================="
#  Parameters and constants
"===================================="

MHz = 1E6; μm = 1E-6;
Nions = 4; z0 = 20μm; d = 2*z0/Nions; d += d/(Nions -1);

### Fitting F = ax + bx^3 ###
#a,b = -8.1389*10^-14, -0.000863192 
a,b = -5.46492*10^-15, -3.486*10^-6; #4 ions, 40 μm spacing
a,b = -4.37193*10^-14, -0.000111552; #4 ions, 20 μm spacing

### Frequencies ###
ωtrap = 2*π*MHz*[1.0, 1.0, 0.2];
ωtrap[3] = abs(√(ee^2/(2π*ϵ0*mYb*d^3)))


"===================================="
# Positions and Hessian for equidistant 1D crystal
"===================================="

pos_ions = PositionIons(Nions,ωtrap,z0,[a,b],plot_position=false)
pos_ions = Chop_Sort(pos_ions)

hess_hom=Hessian(pos_ions, ωtrap, z0, [a,b]; planes=[3])

"========================================"
# Analytical derivatives: Adjoint method
"========================================"

function ∂Uⁱʲ∂A(i,j,U,λ)
    F = [((m!=n)*(λ[m]-λ[n])^-1) for m in 1:length(λ), n in 1:length(λ)]
    Λ̅ = zeros(size(U)) #adjoint of diag(λ)
    U̅ = zeros(size(U)); U̅[i,j] = 1;
    ### Adjoint
    #A̅ = U*(Λ̅  + F.*(U'*U̅))*U'
    A̅ = U*(Λ̅  + 0.5*F.*(U'*U̅ - U̅'*U))*U' #symmetrized
    return A̅
end

function ∂Λⁱ∂A(i,U)
    A̅ = U[:,i] ⊗ U[:,i]
    return A̅
end

"=========================="
# Target function gradient
"=========================="

function ∂Jᵏˡ∂A(k,l,U,λ,μ)
    Θ(i) = (μ^2-λ[i])^-1
    return ∑([Θ(m)*U[l,m]*∂Uⁱʲ∂A(k,m,U,λ) + Θ(m)*U[k,m]*∂Uⁱʲ∂A(l,m,U,λ) + Θ(m)^2*U[k,m]*U[l,m]*∂Λⁱ∂A(m,U) for m in 1:length(λ)])
end

function A̅ₖₗ(k,l,U,λ,μ)
    Θ(m) = (μ^2-λ[m])^-1
    F = [((i!=j)*(λ[i]-λ[j])^-1) for i in 1:length(λ), j in 1:length(λ)]
    Λ̅  = diagm([Θ(i)^2*U[k,i]*U[l,i] for i=1:length(λ)])
    U̅ = zeros(length(λ), length(λ))
    U̅ = [(k==i || l==i) && sum([Θ(j)*U[n,j] for n=1:length(λ)]) for i=1:length(λ), j=1:length(λ)]
    A̅ = U*(Λ̅  + 0.5*F.*(U'*U̅ - U̅'*U))*U'
    return A̅
end


function A̅crₖₗ(k, l, U, λ, μ)
    dim = length(λ);
    Θ(i) = (μ^2-λ[i])^-1
    F = [((i!=j)*(λ[i]-λ[j])^-1) for i in 1:dim, j in 1:dim]
    Λ̅ = diagm([Θ(i)^2*U[k,i]*U[l,i] for i=1:dim])
    U̅ = [(k==i || l==i) && sum([Θ(j)*U[n,j] for n=1:dim]) for i=1:dim, j=1:dim]
    #dΛdA = diagm(diag(U'*ones(length(λ), length(λ))*U))
    A̅ = sum([U̅[i,j]*∂Uⁱʲ∂A(i,j,U,λ) for i=1:dim for j=1:dim]) + sum([Λ̅[i,i]*∂Λⁱ∂A(i,U) for i=1:dim])
    return A̅
end


"============================"
# Unfeasibility of solution
"============================"

using JuMP, Clp


function GradientConstrains(targetArray, unpinnedHessian, threshold)
    dims = size(targetArray)[2];
    Jₜ = targetArray; J₀ = unpinnedHessian;
    number_constraints = Int(dims*(dims-1)/2)
    ΔJ = Jₜ - J₀
    n = 1;
    C = zeros(number_constraints, dims) 
    for k = 1:dims, l = k+1:dims
        n += 1
        A̅ = A̅ₖₗ(k,l,U,λ,μ)
        ∂Jᵏˡ∂Ωⁱⁱ(i) = A̅[i,i]
        C[n,:] = sign(ΔJ[k,l])*([∂Jᵏˡ∂Ωⁱⁱ(i) for i=1:dim]) + threshold
    end
    return C
end

model = Model(Clp.Optimizer)
@variable(model, Ω[1:Nions])
@constraint(model, con[i = 1:Nions], C * Ω .>= 0)


"======="
# Plots
"======="

λ0, b0 = eigen(hess_hom); ω0 = sqrt.(λ0)*ωtrap[3]
δ0 = 2π*0.05MHz; #detuning
μ0 = ω0[1] - δ0

λm, bm = eigen(hess_hom + diagm([4., 4., 0., 0.]))

bm

∂Jᵏˡ∂A(1, 2, b0, λ0, μ0)
A̅ₖₗ(1, 2, b0, λ0, μ0/ωtrap[3])
A̅crₖₗ(1, 2, b0, λ0, μ0/ωtrap[3]) 



i=1;j=4
PlotAnyMatrix(diagm(diag(∂Jᵏˡ∂A(i,j,b0,λ0,μnote))),label_x="i", label_y="j", label_c=string("∂Jⁱʲ/∂A, i,j=",i,",",j));gcf()
PlotAnyMatrix(∂Jᵏˡ∂A(2,1,b0,λ0,μnote))

diagpin = zeros(6,51,4)

m=6
i=3;j=4
μn = μ0
for n in 0:50
    μn += 2π*0.005*MHz
    prod(2π*0.005*MHz .< abs.(μn .-ω0)) && (diagpin[m,n+1,:]= diag(A̅ₖₗ(i, j, b0, λ0, μn/ωtrap[3])))
end

Jij=["\$J^{(1,2)}\$","\$J^{(1,3)}\$","\$J^{(1,4)}\$","\$J^{(2,3)}\$","\$J^{(2,4)}\$","\$J^{(3,4)}\$"]
colorBuGn = PyPlot.cm.jet(range(0, 1, length=6))
markerP=["x", "+", "o","x", "+", "o"]

format_Plot.MultiPlot()

fig, ax = plt.subplots()
for i=1:6
    ax.plot(1:51, diagpin[i,:,4], label=Jij[i], color=colorBuGn[i,:], markersize=6, marker=markerP[i], fillstyle= "none",linestyle="-.",linewidth=0.5);
    ax.set_yscale("symlog")
    ax.set_ylabel("\$ \\log \\partial J^{(i,j)}/ \\partial A \$")
    ax.set_xlabel("\$ \\mu \$")
    ax.set_title("\$ \\Omega_4 \\neq 0 \$")
end
ax.legend(fontsize=20)
gcf()



PlotGradient(diagpin)


function PlotGradient(gradArray)
    fig, ax = plt.subplots()
    ax.matshow(Array((sign.(gradArray).*log.(abs.(gradArray)))'));
    #fig.colorbar(mat,orientation="horizontal",ax=ax,shrink=0.5);
    title("\$ \\partial J^{(i,j)}/ \\partial A \\;\\text{where i,j=}\$"*string(i,",",j));
    ylabel("pinned ion");
    xlabel("\$\\mu\$(a.u.)")
    ax.xaxis.tick_bottom()
    ax.set_yticklabels([1,1,2,3,4]);
    gcf()
end






norm(aa) == √tr(aa'*aa)

μnote1


i=6
PlotAnyMatrix(diagless(∂J₀∂Aₖₗ(i,i,μnote)), label_x="i", label_y="j", label_c="∂Jⁱʲ/∂Aⁱⁱ, i="*string(i));gcf()


function LeadingPin(k,l,μ,type)
    ∂Jᵏˡ∂Aⁱⁱ=[∂J₀∂Aₖₗ(i,i,μ)[k,l] for i=1:Nions]
    
    if type == "abs"
        findmax(∂Jᵏˡ∂Aⁱⁱ)[1] >= abs(findmin(∂Jᵏˡ∂Aⁱⁱ)[1]) ? (return findmax(∂Jᵏˡ∂Aⁱⁱ)) : (return findmin(∂Jᵏˡ∂Aⁱⁱ));
    else
        type=="max" && return findmax(∂Jᵏˡ∂Aⁱⁱ)
        type=="min" && return findmin(∂Jᵏˡ∂Aⁱⁱ)
    end
end

maxPin=diagless([LeadingPin(k,l,μnote,"max")[2] for k=1:Nions, l=1:Nions])
maxVal=diagless([LeadingPin(k,l,μnote,"max")[1] for k=1:Nions, l=1:Nions])

minPin=diagless([LeadingPin(k,l,μnote,"min")[2] for k=1:Nions, l=1:Nions])
minVal=diagless([LeadingPin(k,l,μnote,"min")[1] for k=1:Nions, l=1:Nions])

absPin=diagless([LeadingPin(k,l,μnote,"abs")[2] for k=1:Nions, l=1:Nions])
absVal=diagless([LeadingPin(k,l,μnote,"abs")[1] for k=1:Nions, l=1:Nions])


PlotAnyMatrixL(maxVal, label_x="i", label_y="j", label_c="max(∂Jⁱʲ/∂Aⁱⁱ)", label_vals=maxPin)
gcf()

PlotAnyMatrixL(minVal, label_x="i", label_y="j", label_c="min(∂Jⁱʲ/∂Aⁱⁱ)", label_vals=minPin)
gcf()

PlotAnyMatrixL(absVal, label_x="i", label_y="j", label_c="abs(∂Jⁱʲ/∂Aⁱⁱ)", label_vals=absPin)
gcf()

maximum([∂J₀∂Aₖₗ(i,i,Hom1D[:μnote])[k,l] for i=1:Nions])





diagless(A) = A - diagm(diag(A))


n=12;
PlotAnyMatrix(dtotn(n), label_x="Aⁱⁱ", label_y="bʲₘ", label_c="∂bʲₘ/∂Aⁱⁱ, m="*string(n));
gcf()


PlotAnyMatrix(bm, label_x="bₘ", label_y="bₘⁱ",comment="ωₘ= "*string(round.(λm,digits=3)))
gcf()



λ0, b0= eigen(hess);

db₀dAₖₗ(m,k,l) = sum(((m!=n)*(λ0[m]-λ0[n])^-1)*b0[k,m]*b0[l,n]*b0[:,n] for n=1:Nions)

dtot0(n) = reshape(vcat([db₀dAₖₗ(n,i,i) for i=1:Nions]...),Nions,Nions)

n=12;
PlotAnyMatrix(dtot0(n), label_x="Aⁱⁱ", label_y="bʲₘ", label_c="∂bʲₘ/∂Aⁱⁱ, m="*string(n));
gcf()

PlotAnyMatrix(b0, label_x="bₘ", label_y="bₘⁱ",comment="ωₘ= "*string(round.(λ0,digits=3)))
gcf()


JPL(α) = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]

ll, bb = eigen(JPL(3))

round.(bb'*JPL(3)*bb, digits=4)




"=================="
# Derivatives
"=================="

### Phonon mode

dbₘdAₖₗ(m,k,l) = Σ(((m!=n)*(λm[m]-λm[n])^-1)*bm[k,m]*bm[l,n]*bm[:,n] for n=1:Nions)
db₀dAₖₗ(m,k,l) = Σ(((m!=n)*(λ0[m]-λ0[n])^-1)*b0[k,m]*b0[l,n]*b0[:,n] for n=1:Nions)

dtotn(n) = reshape(vcat([dbₘdAₖₗ(n,i,i) for i=1:Nions]...),Nions,Nions)
dtot0(n) = reshape(vcat([db₀dAₖₗ(n,i,i) for i=1:Nions]...),Nions,Nions)


### Phonon frequency

dλₘdAᵏˡ(m,k,l) = bm[k,m]*bm[l,m]
dλ₀dAᵏˡ(m,k,l) = b0[k,m]*b0[l,m]


### Coupling matrix

Θ(m, μ) = (μ^2-λm[m])^-1
Θ₀(m, μ) = (μ^2-λ0[m])^-1

∂J∂Aₖₗ(k,l,μ) = Σ(Θ(m,μ)*(dbₘdAₖₗ(m,k,l) ⨶ bm[:,m] + bm[:,m] ⨶ dbₘdAₖₗ(m,k,l)) for m=1:Nions) + Σ(Θ(m,μ)^2*dλₘdAᵏˡ(m,k,l)*(bm[:,m] ⨶ bm[:,m]) for m=1:Nions)

∂J₀∂Aₖₗ(k,l,μ) = Σ(Θ₀(m,μ)*(db₀dAₖₗ(m,k,l) ⨶ b0[:,m] + b0[:,m] ⨶ db₀dAₖₗ(m,k,l)) for m=1:Nions) + Σ(Θ₀(m,μ)^2*dλ₀dAᵏˡ(m,k,l)*(b0[:,m] ⨶ b0[:,m]) for m=1:Nions)