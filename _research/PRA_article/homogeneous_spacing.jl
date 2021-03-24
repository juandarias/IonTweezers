using DrWatson, NLopt, Optim
push!(LOAD_PATH, srcdir());
include(srcdir()*"/constants.jl");
using optimizers, coupling_matrix, crystal_geometry



"===================================="
# Potential for equidistant 1D crystal
"===================================="

### Ion (z)-coordinates and forces ###
prefactor = ee^2/(4*π*ϵ0);  
Nions = 12; z0 = 40*10^-6; d = 2*z0/Nions; d += d/(Nions -1);
zᵢ(i) = -z0+(i-1)*d;
fzᵢ(i) = prefactor * sum([i==j ? 0 : (zᵢ(j)-zᵢ(i))/abs((zᵢ(i)-zᵢ(j))^3) for j=1:Nions]);
zdata = [zᵢ(i) for i=1:Nions]; fzdata = [fzᵢ(i) for i=1:Nions];

### Fitting F = ax + bx^3 ###
a,b = 1.84*10^-14, -0.000110607

### Frequencies ###
ω_trap = 2*pi*1E6*[0.6, 4.0, 0.2];
ω_trap[3] = abs(√(ee^2/(2π*ϵ0*mYb*d^3)))


"===================================="
# Positions for equidistant 1D crystal
"===================================="

pos_ions = PositionIons(Nions,ω_trap,z0,[a,b],plot_position=false)
pos_ions= round.(pos_ions, digits=7)


pos_ionss=sort(pos_ions,dims=2)

Ωpin1D= ω_trap[3]*sqrt.([0.202449,0.195256,0.054453,0.0748212,0.0937544,0.23156,0.23156,0.0937544,0.0748212,0.054453,0.195256,0.202449]);

Jexp, evals, evecs = Jexp1Dm(pos_ionss, Ωpin1D, ω_trap, 2.76346*ω_trap[3], equidistant=true, coeffs_field=[a,b], size_crystal=z0)

evals

(evals/ω_trap[3]).^2

{0.00347471,{0.202449,0.195256,0.054453,0.0748212,0.0937544,0.23156,0.23156,0.0937544,0.0748212,0.054453,0.195256,0.202449},2.76346,0.296715}






@btime Hessian(pos_ions[:,1:4], ω_trap, [a,b], z0)
@btime Hessian(pos_ions, ω_trap, [a,b], z0, planes=[1,2,3])


using LinearAlgebra, Arpack, BenchmarkTools

rn = rand(10,10);
rn3 = rand(30,30);

@btime eigen(rn)

Jexp1Dm(pos_ions, rand(12), ω_trap, 1.5; equidistant=true, coeffs_field=[a,b], size_crystal=z0)


kron(diagm([1,0,0]), diagm(([1.0,2.0,3.0]).^2))

diagm([sign([1.0,2.0,3.0][i]) for i in 1:3])













function SecondDerivative()
    dir = [Sym("x"),Sym("y"),Sym("z")];
    Vtrap(x,y,z) = a*x^2/(4*e) - b*x^4/(16*e) - b*y^4/(16*e) - b*z^4/(4*e) + y^2*(2*a - 3*b*x^2)/(8*e) + z^2*(-2*a + 3*b*x^2 + 3*b*y^2)/(4*e)
    ddVtrap = [diff(Vtrap(x,y,z), dir[α], dir[β]) for α=1:3, β=1:3]
end


diff(Vtrap(x,y,z),x,y)



fcn_as_string = "sin.(2*pi*x*y)" 
fcn = eval(Meta.parse("(x,y) -> " * fcn_as_string))
@show fcn(1.0,2.0)

fcn(3.0,2.0)-sin(2*π*6)


"===================================="
# Potential for equidistant 1D crystal
"===================================="






#GD test

function GDPin(Nions::Int, fun::Function, xguess::Vector, Ωmax::Float64; μmax::Float64=0.0,  μmin::Float64=0.0, c1max::Float64=0.0, c1min::Float64=0.0, show_trace::Bool=false, kwargs...)

    #Bounds
    #numPars = length(xguess);
    lc = append!(zeros(Nions), [μmin, c1min]);
    uc = append!(Ωmax*ones(Nions), [μmax, c1max]);

    #Optimizer
    inner_optimizer = GradientDescent()
    res = Optim.optimize(fun, lc, uc, xguess, Fminbox(inner_optimizer), Optim.Options(;kwargs...))

    return res
end


#Working optimization and fits#
function potential(p::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = sum(2*p[1]*zdata.^2 + 2*p[2]*zdata.^4-2*zdata.*fzdata)
        grad[2] = sum(2*p[2]*zdata.^6 + 2*p[1]*zdata.^4-2*fzdata.*zdata.^3)
    end
    return sum((p[1]*zdata + p[2]*(zdata.^3) - fzdata).^2);
end

opt = Opt(:LD_MMA, 2)
opt.min_objective = potential
(minf,minx,ret)= NLopt.optimize(opt, [-0.01,-0.00052])

#using polynomial# 
fit_cubic = fit(zdata,fz,3) # degree = length(xs) - 1
fit_cubic.coeffs






























#Non-working optimizers#

m(t,p) = -p[1]*t+-p[2]*t.^3; p0= [-0.5,0.5];

using LsqFit
lsq_fit_cubic = curve_fit(m,zdata,fzdata,p0; autodiff=:forwarddiff)

coef(lsq_fit_cubic)




f(p) = sum((p[1]*zdata + p[2]*(zdata.^3) - fzdata).^2);
p0 = [-0.05,-0.00052]

soln_op= optimize(f,p0)
soln_op.minimizer


function potential(p::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = sum(2*p[1]*zdata.^2 + 2*p[2]*zdata.^4-2*zdata.*fzdata)
        grad[2] = sum(2*p[2]*zdata.^6 + 2*p[1]*zdata.^4-2*fzdata.*zdata.^3)
    end
    return sum((p[1]*zdata + p[2]*(zdata.^3) - fzdata).^2);
end


function nlopt_form(f,pp,gg)
    if length(gg) > 0
        df = DiffResults.GradientResult(pp)
        df = ForwardDiff.gradient!(df,f,pp)
        gg .= DiffResults.gradient(df)
        return DiffResults.value(df)
    else
        return sum((pp[1]*zdata + pp[2]*(zdata.^3) - fzdata).^2);
    end
end


opt = Opt(:LN_COBYLA, 2)
opt.min_objective = potential
min_objective!(opt, (x,grad) -> nlopt_form(f,x,grad))
#opt = Opt(:AUGLAG, 2) # algorithm (see NLopt Algorithms for possible values) and the dimensionality of the problem
#lopt = Opt(:LD_LBFGS, 2)
#local_optimizer!(opt,lopt)




