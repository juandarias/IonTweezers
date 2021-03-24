module optimizers

using Optim, NLSolversBase, LineSearches#, NLopt;

export IPNewtonPin, MMAPin, AUGLAGPin, GDPin, CGDPin, GDPin4;

"============================================"
# Optim: IPNewton with a constraint function
"============================================"


    function IPNewtonPin(xguess::Vector, Ωmax::Float64, Nions::Int, fun, grad, hess, con, con_jacob, con_hes; μmax::Float64=0.0, c1max::Float64=0.0)
        #Bounds
        lx = Float64[]; ux = Float64[];
        numPars = length(xguess)
        lc =zeros(numPars)
        μmax!=0 && c1max!=0 ? uc = append!(Ωmax*ones(Nions),[μmax, c1max, Ωmax]) : μmax!=0 ? uc = append!(Ωmax*ones(Nions),[μmax, Ωmax]) : c1max!=0 ? uc = append!(Ωmax*ones(Nions),[c1max, Ωmax]) : uc = Ωmax*ones(Nions)

        df = TwiceDifferentiable(fun, grad, hess, xguess)
        dfc = TwiceDifferentiableConstraints(con, con_jacob, con_hes, lx, ux, lc, uc)
        res = Optim.optimize(df, dfc, xguess, IPNewton())
        return res
    end



"========================================="
# NLopt: Method-of-moving-asymptotes (MMA)
"========================================="

    function MMAPin(xguess::Vector, Ωmax::Float64, Nions::Int, fun::Function,cons::Function; μmax::Float64=0.0, c1max::Float64=0.0)
        numPars = length(xguess)
        opt = Opt(:LD_MMA, numPars) # algorithm (see NLopt Algorithms for possible values) and the dimensionality of the problem
        min_objective!(opt,(Ω,g) -> fun(Ω,g))
        inequality_constraint!(opt, (Ω,g) -> cons(Ω,g,Ωmax))
        ftol_rel!(opt,1e-9)

        # Bounds of each variable
        lower_bounds!(opt, zeros(numPars))
        μmax!=0 && c1max!=0 ? upper_bounds!(opt, append!(Ωmax*ones(Nions),[μmax, c1max, Ωmax])) : μmax!=0 ? upper_bounds!(opt, append!(Ωmax*ones(Nions),[μmax, Ωmax])) : c1max!=0 ? upper_bounds!(opt, append!(Ωmax*ones(Nions),[c1max, Ωmax])) : upper_bounds!(opt, Ωmax*ones(Nions))

        NLopt.optimize(opt, xguess) #algorithm, initial guess
    end


"============================="
# NLopt: Augmented Lagrangian: has to be updated following MMA example
"============================="

    function AUGLAGPin(xguess::Vector, Ωmax::Float64, Nions::Int, fun::Function, cons::Function)
        opt = Opt(:AUGLAG, Nions) # algorithm (see NLopt Algorithms for possible values) and the dimensionality of the problem
        lopt = Opt(:LD_LBFGS, Nions)
        local_optimizer!(opt,lopt) #Don't forget to set a stopping tolerance for this subsidiary optimizer!
        lower_bounds!(opt, zeros(Nions))
        min_objective!(opt,(Ω,g) -> fun(Ω,g))
        inequality_constraint!(opt, (Ω,g) -> cons(Ω,g,Ωmax))
        ftol_rel!(opt,1e-9)
        ftol_rel!(lopt,1e-9)
        NLopt.optimize(opt, xguess) #algorithm, initial guess
    end

end
