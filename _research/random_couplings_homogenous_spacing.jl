using DrWatson, PyPlot
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using optimizers, coupling_matrix, crystal_geometry, plotting_functions



"========================="
# Auxiliary functions
"========================="

Chop_Sort(x) = sort(round.(x,digits=5), dims=2)
diagless(A) = A - diagm(diag(A))
Σ = sum
⨷(A,B) = kron(A',B)

"===================================="
#  Parameters and constants
"===================================="

MHz = 1E6; μm = 1E-6;
Nions = 6; z0 = 18*10^-6; d = 2*z0/Nions; d += d/(Nions -1);

### Fitting F = ax + bx^3 ###
a,b = -8.1389*10^-14, -0.000863192

### Frequencies ###
ω_trap = 2*pi*1E6*[0.6, 4.0, 0.2];
ω_trap[3] = abs(√(ee^2/(2π*ϵ0*mYb*d^3)))
μnote = 0.8132 #2π 100 kHz red detuned from first phonon mode


"===================================="
# Positions and Hessian for equidistant 1D crystal
"===================================="

pos_ions = PositionIons(Nions,ω_trap,z0,[a,b],plot_position=false)
pos_ions = Chop_Sort(pos_ions)

hess=Hessian(pos_ions, ω_trap, z0, [a,b]; planes=[1]);
λ0, b0= eigen(hess);


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

μnote

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
