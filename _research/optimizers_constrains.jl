using Optim, NLSolversBase, LineSearches #hide




"=========================================="
# Optim: Box Constrained Gradient Descent
"=========================================="

    f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    
    function g!(G, x)
        G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
        G[2] = 200.0 * (x[2] - x[1]^2)
    end

    lower = [1.25, -2.1]
    upper = [Inf, Inf]
    initial_x = [2.0, 2.0]
    inner_optimizer = GradientDescent()
    results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))

    ### Using Line search
    lower = [1.25, -2.1]
    upper = [Inf, Inf]
    initial_x = [2.0, 2.0]
    # requires using LineSearches
    inner_optimizer = GradientDescent(linesearch=LineSearches.BackTracking(order=3))
    results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))

    "  Candidate solution "
    #Minimizer: [1.25e+00, 1.56e+00]
    #Minimum:   6.250000e-02
    "  Candidate solution "



"=========="
# Optim: IPNewton with box constraints
"=========="

    fun(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

    function fun_grad!(g, x)
    g[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    g[2] = 200.0 * (x[2] - x[1]^2)
    end

    function fun_hess!(h, x) 
    h[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    h[1, 2] = -400.0 * x[1]
    h[2, 1] = -400.0 * x[1]
    h[2, 2] = 200.0
    end;

    ### Box minimization

    x0 = [2.0, 2.0]
    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)

    #lx = [-0.5, -0.5]; ux = [0.5, 0.5]
    lx = [1.25, -2.1]; ux = [Inf, Inf]
    dfc = TwiceDifferentiableConstraints(lx, ux)

    res = optimize(df, dfc, x0, IPNewton())

    " Candidate solution"
    #Minimizer: [1.25e+00, 1.56e+00]
    #Minimum:   6.250000e-02
    " Candidate solution"



"============================================"
# Optim: IPNewton with a constraint function
"============================================"
    con_c!(c, x) = (c[1] = x[1] + x[2]; c)
    function con_jacobian!(J, x)
        J[1,1] = 1
        J[1,2] = 1
        J
    end 
    
    ### What if Hessian of box constraint is zero?
    
    function con_h!(h, x, λ)
        h[1,1] += 0
        h[2,2] += 0
    end;

    x0 = [0.0, 0.0]
    lx = Float64[]; ux = Float64[]
    lc = [-Inf]; uc = [0.5^2]
 
    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)

    dfc = TwiceDifferentiableConstraints(con_c!, con_jacobian!, con_h!,
                                     lx, ux, lc, uc)
    res = optimize(df, dfc, x0, IPNewton())

    "Candidate solution"
    
    #Minimizer: [4.56e-01, 2.06e-01]
    #Minimum:   2.966216e-01
    "Candidate solution"
    
    "Candidate solution linear constraint"
    #Minimizer: [2.11e-01, 3.90e-02]
    #Minimum:   6.255691e-01
    "Candidate solution linear constraint"
    




"========================================================="
# NLopt: using method-of-moving-asymptotes (MMA) and Augmented lagragnian
# https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#mma-method-of-moving-asymptotes-and-ccsa
"========================================================="

using NLopt

function rosenbrockf(x::Vector,grad::Vector)
    if length(grad) > 0
	    grad[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
	    grad[2] = 200.0 * (x[2] - x[1]^2)
    end
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end
function r_constraint(x::Vector, grad::Vector)
    if length(grad) > 0
	grad[1] = 1
	grad[2] = 1
	end
	return x[1] + x[2] - 0.25
end
opt = Opt(:LD_MMA, 2) # algorithm (see NLopt Algorithms for possible values) and the dimensionality of the problem 
opt = Opt(:AUGLAG, 2) # algorithm (see NLopt Algorithms for possible values) and the dimensionality of the problem
lopt = Opt(:LD_LBFGS,2)
local_optimizer!(opt,lopt) #Needed for augmented lagragnian
lower_bounds!(opt, [-5, -5.0])
min_objective!(opt,(x,g) -> rosenbrockf(x,g))
inequality_constraint!(opt, (x,g) -> r_constraint(x,g))
ftol_rel!(opt,1e-9)
NLopt.optimize(opt, [0.0,0.0]) #algorithm, initial guess

"Candidate solution MMA"
# (0.2966215685276063, [0.45564896272703836, 0.20587380402969943], :FTOL_REACHED)
# linear constraint (0.6255690535411272, [0.21101920212532416, 0.038980811922948236], :FTOL_REACHED)
"Candidate solution MMA"

"Candidate solution AUGLAG"
# (0.29662156969049963, [0.4556489634089007, 0.20587379918309445], :FTOL_REACHED)
"Candidate solution AUGLAG"

"======================"
# Lagrange multipliers
"======================"

gr()
f(x1,x2) = -exp.(-(x1.*x2 - 3/2).^2 - (x2-3/2).^2)
c(x1) = sqrt(x1)
x=0:0.01:3.5
contour(x,x,(x,y)->f(x,y),lw=1.5,levels=[collect(0:-0.1:-0.85)...,-0.887,-0.95,-1])
plot!(c,0.01,3.5,label="",lw=2,color=:black)
scatter!([1.358],[1.165],markersize=5,markercolor=:red,label="Constr. Optimum")



HS9 = MultivariateProblems.ConstrainedProblems.examples["HS9"]