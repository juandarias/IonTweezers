using Optim, LineSearches;
using LinearAlgebra, PyPlot

"=================="
# Gradient descent 
"=================="

    f(x) = -sin(x[1]^2/2 - x[2]^2/4 + 3) * cos(2x[1] + 1 - exp(x[2]))

    function fplot(f)
        x = range(-2.5, stop=2.5, length=200)
        y = range( -1.5, stop=1.5, length=100)
        u = [f([x,y]) for y in y, x in x]
        
        gcf().set_size_inches(12,5)
        contour(x, y, u, 25, colors="k", linewidths=0.5)
        contourf(x, y, u, 25)
        axis("equal")
        colorbar()
        return
    end

    fplot(f)

    show()


    ### Gradient function ###

    function df(x)
        a1 = x[1]^2/2 - x[2]^2/4 + 3
        a2 = 2x[1] + 1 - exp(x[2])
        b1 = cos(a1)*cos(a2)
        b2 = sin(a1)*sin(a2)
        return -[x[1]*b1 - 2b2, -x[2]/2*b1 + exp(x[2])*b2]
    end


    ### Gradient descent function ###

    function gradient_descent(f, df, x0, α=0.1; tol=1e-4, maxiter=500)
        x = x0
        xs = [x0]
        for i = 1:maxiter
            gradient = df(x)
            if sqrt(sum(gradient.^2)) < tol
                break
            end
            x -= α*gradient
            push!(xs, x)
        end
        return xs
    end

    x0s = [[-2,.5], [0,.5], [2.2,-0.5]]; #Initial guesss

    for x0 in x0s
        xs = gradient_descent(f, df, x0, 0.3)
        println("Path length = $(length(xs)), ||gradient|| = $(sqrt(sum(df(xs[end]).^2)))")
    end


"=================================="
# Gradient descent with line search
"=================================="


    ### Line search ###

    function line_search(f, direction, x, αmin=1/2^20, αmax=2^20)
        α = αmin
        fold = f(x)
        while true
            if α ≥ αmax
                return α
            end
            fnew = f(x - α*direction)
            if fnew ≥ fold
                return α/2
            else
                fold = fnew
            end
            α *= 2
        end
    end

    ### Line search ###

    function gradient_descent_line_search(f, df, x0; tol=1e-4, maxiter=500)
        x = x0
        xs = [x0]
        for i = 1:maxiter
            gradient = df(x)
            if sqrt(sum(gradient.^2)) < tol
                break
            end
            α = line_search(f, gradient, x)
            x -= α*gradient
            push!(xs, x)
        end
        return xs
    end


    x0s = [[-2,.5], [0,.5], [2.2,-0.5]]; #Initial guess


    for x0 in x0s
        xs = gradient_descent_line_search(f, df, x0)
        #plot_path(xs)
        println("Path length = $(length(xs)), ||gradient|| = $(sqrt(sum(df(xs[end]).^2)))")
    end

"================="
# Newton method #
"================="

    function finite_difference_hessian(f, x, ϵ=1e-5)
        n = length(x)
        ddf = zeros(n,n)
        f0 = f(x)
        for i = 1:n
            x1 = copy(x)
            x1[i] += ϵ
            fP = f(x1)
            x1[i] -= 2ϵ
            fM = f(x1)
            ddf[i,i] = (fP - 2*f0 + fM) / ϵ^2
        end
        for i = 1:n, j = i+1:n
            x1 = copy(x)
            x1[i] += ϵ
            x1[j] += ϵ
            fPP = f(x1)
            x1[i] -= 2ϵ
            fMP = f(x1)
            x1[j] -= 2ϵ
            fMM = f(x1)
            x1[i] += 2ϵ
            fPM = f(x1)
            ddf[i,j] = (fPP - fMP - fPM + fMM) / 4ϵ^2
            ddf[j,i] = ddf[i,j]
        end
        return ddf
    end


"================================"
# Newton method with Line search
"================================"

    rosenbrock = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]
    
    algo_hz = Optim.Newton(linesearch = HagerZhang())    # Both Optim.jl and IntervalRootFinding.jl export `Newton`
    res_hz = Optim.optimize(rosenbrock.f, rosenbrock.g!, rosenbrock.h!, rosenbrock.initial_x, method=algo_hz)
    
    optimize(rosenbrock.f, rosenbrock.g!, rosenbrock.h!, [0.0, 0.0], Optim.Newton(),Optim.Options(show_trace=true))
    "============================="
    # Gradient descent using Optim
    "============================="
    
    GD = optimize(rosenbrock.f, rosenbrock.g!,[0.0, 0.0],GradientDescent())
    rosenbrock = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]
