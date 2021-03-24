using DrWatson, Distributions, Random, LinearAlgebra
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using optimizers, coupling_matrix, crystal_geometry, plotting_functions



"========================="
# Parameters and constants
"========================="

MHz = 1E6; μm = 1E-6;
Nions = 7; ωtrap = 2π*MHz*[1, 0.4, 0.4];

Chop_Sort2D(A) = A[:, sortperm(A[3,:])]
diagless(A) = A - diagm(diag(A))

"========================="
# Positions and Hessian
"========================="

pos_ions = PositionIons(Nions,ωtrap,plot_position=false);
pos_ions=Chop_Sort2D(pos_ions)

hess=Hessian(pos_ions, ωtrap;planes=[1]);
λx, bx= eigen(hess);


"========================="
# Random target matrix
"========================="

# Couplings
edges_FM=collect(Iterators.product(4,[1 2 3 5 6 7]))

# Random distribution with mean j₀
Jᵣ(j₀) = truncated(Normal(j₀, 0.95), 0.0, Inf)

# Random star matrix
Jstar = :(Jtarget2(edges_FM,[],Nions,J_strength=rand(Jᵣ(1),7)))

# Homogeneous matrix
JstarFM = Array(Jtarget2(edges_FM,[],Nions))

# Experimental matrix
Jexp(parms)=Jexp1D(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+2], hessian=hess)[1];
#Jexp(parms)=Jexp1D(pos_ions, parms[1:Nions], ωtrap, parms[Nions+1], hessian=hess)[1];
λexp(parms)=Jexp1D(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+2], hessian=hess)[2];


"========================="
# Objective function and constrains
"========================="

# Seed and constrains
Ωpin(parms) = vcat(parms[1:Nions÷2], parms[4], reverse(parms[1:Nions÷2]))
Ω₄=0.1; μ₀ = 1.0;
parmstar₀ = vcat(zeros(Nions÷2),Ω₄, μ₀);
parms₀ = vcat(Ωpin(parms₀), parms₀[5])
Ωmax = 5.0; μmin = 0.5; μmax = 20;
lc = append!(-0.001*ones(Nions÷2+1), μmin);
lc = append!(-0.001*ones(Nions), μmin);
uc = append!(Ωmax*ones(Nions÷2+1), μmax);
uc = append!(Ωmax*ones(Nions), μmax);

# Objective functions
#DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./(abs(coupling_matrix[1,2]))
DividebyCross(coupling_matrix::Array{Float64,2}) = coupling_matrix./abs(sum(coupling_matrix[1:3,4]+coupling_matrix[4,1:3])/6) #uses average of center to edge couplings to normalize experimental matrix

#ϵ_J1(parms) = norm(DividebyMax(Jexp(parms)) - eval(Jstar))
#ϵ_NormNorm(parms) = norm(normalize(Jexp(parms)) - normalize(eval(Je)))
#ϵ_Cross(parms) = norm(DividebyCross(Jexp(parms)) - eval(Je))
#ϵ_Had(parms) = 1E6*norm(JstarFM.*normalize(Jexp(parms)) + normalize(eval(Je)))
#ϵ_Had(parms) = norm(JstarFM.*DividebyCross(Jexp(parms)) + eval(Je))
ϵ_NormNorm(parms) = norm(normalize(Jexp(parms)) - normalize(JstarFM))
ϵ_Cross(parms) = norm(DividebyCross(Jexp(parms)) - JstarFM)
ϵs_star = (ϵ_NormNorm, ϵ_Cross)

JstarFM.*eval(Je)

"========================="
# Optimization and Benchmarking
"========================="

# Algorithms
line_search = (HagerZhang(), BackTracking(), BackTracking(order=2));
algorithm(lsa) = (LBFGS(linesearch = lsa), BFGS(linesearch = lsa), GradientDescent(linesearch = lsa), ConjugateGradient(linesearch = lsa))

solution2 = Optim.optimize(ϵ_NormNorm, lc, uc, parmstar₀, Fminbox(algorithm(BackTracking(order=3))[1]), Optim.Options(time_limit = 500.0, store_trace=true, allow_f_increases=true, g_tol=1e-24))

round.(DividebyCross(Jexp(solution2.minimizer)),digits=3)

λexp(solution2.minimizer)

a=1;
star_LBFGS_summmary = [];
star_LBFGS_solution = [];
for n=1:1
    for ϵobj in ϵs_star, lsa in line_search[2:3]
        try 
            solution = Optim.optimize(ϵobj, lc, uc, parms₀, Fminbox(algorithm(lsa)[a]), Optim.Options(time_limit = 500.0, store_trace=true))
        catch error
            println(error)
        end
        Jmin = Jexp(solution.minimizer);
        summary_sol = Dict(:target => "homogenous_star", :method => summary(solution), :objective => string(ϵobj), :linesearch => line_search[l], :minimizer => Array(solution.minimizer), :minimum => solution.minimum, :iterations => solution.iterations, :time => solution.time_run, :Jexp => Array(Jmin))
        push!(star_LBFGS_summmary, summary_sol)
        push!(star_LBFGS_solution, solution)
    end
end
