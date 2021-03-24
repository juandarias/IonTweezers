using DrWatson, PyPlot, LinearAlgebra, Optim, LineSearches, FiniteDifferences
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using optimizers, coupling_matrix, crystal_geometry, plotting_functions


"============="
# Definitions
"============="

MHz = 1E6; μm = 1E-6;


"========================="
# Auxiliary functions
"========================="

Chop_Sort(x) = sort(round.(x,digits=5), dims=2)
Chop_Sort2D(A) = A[:, sortperm(A[3,:])]
diagless(A) = A - diagm(diag(A))
Σ = sum
⊗(A,B) = kron(A',B)


"===================================="
#  Parameters and constants
"===================================="

    Nions = 7;
    ### Frequencies ###
    ωtrap = 2π*MHz*[0.6, 0.6, 0.1];

"========================="
# Positions and Hessian
"========================="

    PositionIons(Nions, ωtrap, plot_position=true, tcool=500E-6, cvel=20E-20)
    pos_ions = Chop_Sort(PositionIons(Nions, ωtrap, plot_position=false, tcool=500E-6, cvel=10E-20));
    hess=Hessian(pos_ions, ωtrap; planes=[1])


"========================="
# Target and experimental matrices
"========================="

    # Target and experimental matrices and phonon frequencies
    JPL(α) = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]
    Jexp(parms) = Jexp1D(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+1], planes=[1], hessian=hess)[1];
    Jexp_no_μ(parms) = Jexp1D(pos_ions, Ωpin(parms), ωtrap, μsol, planes=[1], hessian=hess)[1];
    eigsystem(parms) = Jexp1D(pos_ions, Ωpin(parms), ωtrap, μsol, hessian=hess)[2:3];
    #λp(parms) = Jexp1D(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+1], planes=[1], hessian=hess)[2];
    #Jsoln(solution) = Jexp(solution)./maximum(abs.(Jexp(solution)))

"========================="
# Objective function and constrains
"========================="

    # Seed and constrains
    Ωpin(parms) = vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2]))
    initial_μ = 10.0;
    initial_parms = vcat(0.05*ones(Nions÷2), initial_μ);
    Ωmax = 20.0; μmin = 0.5; μmax = 20;
    lc = append!(zeros(Nions÷2), μmin);
    uc = append!(Ωmax*ones(Nions÷2), μmax);

    # Seed and constrains: no beatnote
    μsol = solution_nograd.minimizer[7]
    Ωpin(parms) = vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2]))
    initial_parms_no_μ = vcat(0.05*ones(Nions÷2));
    Ωmax = 20.0; μmin = 0.5; μmax = 20;
    lc_no_μ = zeros(Nions÷2);
    uc_no_μ = Ωmax*ones(Nions÷2);


    # Objective functions
    DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./(abs(coupling_matrix[1,2]))
    ϵ_J(parms) = norm(DividebyMax(Jexp(parms)) - JPL(α))
    ϵ_J_no_μ(parms) = norm(DividebyMax(Jexp_no_μ(parms)) - JPL(α))
    ϵ_J_sum(parms) = sum(Jexp_no_μ(parms).^2)


    ### Analytical gradient without beatnote
    # Auxiliary functions
    let λ = eigen(hess).values, U = eigen(hess).vectors, μ = μ0;
        global function A̅ₖₗ(k,l)
            dims = length(λ)
            ϵ = 1E-12
            Θ(m) = (μ^2-λ[m])^-1
            F = [((i!=j)*(λ[j]-λ[i])^-1) for i in 1:length(λ), j in 1:length(λ)]
            #F = [(i!=j)*(λ[j]-λ[i])/((λ[j]-λ[i])^2+ϵ) for i in 1:length(λ), j in 1:length(λ)]
            A̅ = zeros(dims,dims)
            for m =1:dims
                Λ̅ = zeros(dims,dims)
                U̅ = zeros(dims, dims)
                Λ̅[m,m] = Θ(m)^2*U[k,m]*U[l,m]
                U̅[k,m] = Θ(m)*U[l,m]
                U̅[l,m] = Θ(m)*U[k,m]        
                A̅ += U*(Λ̅  + 0.5*F.*(U'*U̅ - U̅'*U))*U'
            end
            return A̅
        end
    end



    function A̅ₖₗ(k,l,U,λ)
        dims = length(λ)
        ϵ = 1E-12
        Θ(m) = (μsol^2-λ[m])^-1
        F = [((i!=j)*(λ[j]-λ[i])^-1) for i in 1:length(λ), j in 1:length(λ)]
        #F = [(i!=j)*(λ[j]-λ[i])/((λ[j]-λ[i])^2+ϵ) for i in 1:length(λ), j in 1:length(λ)]
        A̅ = zeros(dims,dims)
        for m =1:dims
            Λ̅ = zeros(dims,dims)
            U̅ = zeros(dims, dims)
            Λ̅[m,m] = Θ(m)^2*U[k,m]*U[l,m]
            U̅[k,m] = Θ(m)*U[l,m]
            U̅[l,m] = Θ(m)*U[k,m]        
            A̅ += U*(Λ̅  + 0.5*F.*(U'*U̅ - U̅'*U))*U'
        end
        return A̅
    end


    function g_no_μ!(G, parms)
        #λ, U = eigsystem(parms)
        ϵⱼ = ϵ_J_no_μ(parms)
        #A̅ₖₗ₂(k,l) = A̅ₖₗ(k,l,U,λ)
        ΔJ = DividebyMax(Jexp_no_μ(parms)) - JPL(α)
        #Do I need to calculate U and λ at each iteration of the gradient? No, see section 2.2.1 of Miles
        ∇ϵ = (ϵⱼ^-1)*sum([ΔJ[i,j]*(A̅ₖₗ(i,j)) for i=1:Nions for j=1:Nions])
        for n=1:length(parms)
            G[n] = ∇ϵ[n,n]
        end
    end


"========================="
# Optimization
"========================="

    α=2.0
    solution_nograd = Optim.optimize(ϵ_J, lc, uc, initial_parms, Fminbox(LBFGS(linesearch=BackTracking())), Optim.Options(time_limit = 500.0, store_trace=true, extended_trace=true))

    solution_nograd.minimum
    solution_nograd.minimizer

    solution_grad = Optim.optimize(ϵ_J_no_μ, g_no_μ!, lc_no_μ, uc_no_μ, initial_parms_no_μ, Fminbox(LBFGS(linesearch=BackTracking())), Optim.Options(time_limit = 500.0))

    #review: torch and adjoint gradients solutions

    solution_grad.minimum
    solution_grad.minimizer


    xs = []; gs = [];
    solution_grad_sum_noevupdate = Optim.optimize(ϵ_J_sum, g_sum! ,lc, uc, initial_parms, Fminbox(LBFGS(linesearch=BackTracking())), Optim.Options(time_limit = 500.0, store_trace=true, extended_trace=true,callback=cb))

    solution_grad_sum_cheap = Optim.optimize(Optim.only_fg!(fg_sum!) ,lc, uc, initial_parms, Fminbox(BFGS(linesearch=BackTracking())), Optim.Options(time_limit = 500.0, store_trace=true, extended_trace=true))

    solution_sum = Optim.optimize(ϵ_J_sum, lc, uc, initial_parms, Fminbox(BFGS(linesearch=BackTracking())), Optim.Options(time_limit = 500.0, store_trace=true, extended_trace=true))

    solution.minimizer #solution with gradient, no update of eigenvectors
    solution_nograd.minimizer
    solution_grad_corr.minimizer #solution with updated gradient function


    ϵ_J_sum(solution_sum.minimizer)
    ϵ_J_sum(solution_grad_sum_noevupdate.minimizer)
    ϵ_J_sum(solution_grad_sum_cheap.minimizer)

"================="
# Plots
"================="

    λ0, b0 = eigen(hessYZ); ω0 = sqrt.(λ0)*ωtrap[3]
    δ0 = 2π*0.05MHz; #detuning
    μ0 = (ω0[1] - δ0)/ωtrap[3]

    let λ = eigen(hessYZ).values, U = eigen(hessYZ).vectors, μ = μ0;
        global function A̅ₖₗ(k,l)
            dims = length(λ)
            ϵ = 1E-12
            Θ(m) = (μ^2-λ[m])^-1
            F = [((i!=j)*(λ[j]-λ[i])^-1) for i in 1:length(λ), j in 1:length(λ)]
            #F = [(i!=j)*(λ[j]-λ[i])/((λ[j]-λ[i])^2+ϵ) for i in 1:length(λ), j in 1:length(λ)]
            A̅ = zeros(dims,dims)
            for m =1:dims
                Λ̅ = zeros(dims,dims)
                U̅ = zeros(dims, dims)
                Λ̅[m,m] = Θ(m)^2*U[k,m]*U[l,m]
                U̅[k,m] = Θ(m)*U[l,m]
                U̅[l,m] = Θ(m)*U[k,m]        
                A̅ += U*(Λ̅  + 0.5*F.*(U'*U̅ - U̅'*U))*U'
            end
            return A̅
        end
    end
    
    i = 6
    PlotAnyMatrix(diagless([A̅ₖₗ(k,l)[i,i] for k=1:Nions, l=1:Nions]));gcf()

    ∇J₆ = diagless([A̅ₖₗ(k,l)[i,i] for k=1:Nions, l=1:Nions])

    ∇J₆norm = DividebyMax(∇J₆)


    norm_cm = PyPlot.colorsm.Normalize(vmin=-1, vmax=1)
    PRGn = PyPlot.cm.PRGn
    cmap = PyPlot.cm.ScalarMappable(norm=norm_cm, cmap=PRGn)

    round.(255*collect(cmap.to_rgba(-1))[1:3], digits=2)


    ExportTikzCouplingGraphA(pos_ions, Jladder, ∇J₆norm, "Ladder_sensitivity", "Article", threshold=0.1)
    
    

    function ExportTikzCouplingGraphA(pos_ions, target_matrix, result_matrix, name, location; threshold=0.05, plane ="YZ")
        Nions = size(pos_ions,2)
    
        ### Exporting vertices
        graph_file = Array{Any,2}(nothing,Nions+1,5)
        header = ["id" "x" "y" "color" "layer"]
        graph_file[1,:]= header
        graph_file[2:end,1] = collect(1:Nions)
        graph_file[2:end,2] = pos_ions[2,:]
        graph_file[2:end,3] = pos_ions[3,:]
        graph_file[2:end,4] .= "blue"
        graph_file[2:end,5] .= 1
        writedlm(datadir(location, name*"vertices.csv"), graph_file, ",")
    
        ### Exporting second layer for residuals
        graph_file[2:end,1] = collect(Nions+1:2*Nions)
        graph_file[2:end,5] .= 2
        writedlm(datadir(location, name*"vertices_residual.csv"), graph_file, ",")
        
    
        ### Exporting edges
        target_edges = result_matrix - result_matrix.*(iszero.(target_matrix))
        residual_edges = result_matrix.*(iszero.(target_matrix))
    
        number_edges=count(i->(i!=0), target_edges)÷2
        edge_file = Array{Any,2}(nothing, number_edges +1,6)
        header = ["u" "v" "R" "G" "B" "label"]
        edge_file[1,:] = header
    
        nn=1
        for i=1:Nions, j=i+1:Nions
            if target_edges[i,j] != 0
                nn+=1
                edge_file[nn,1] = i
                edge_file[nn,2] = j
                edge_file[nn,6] = target_edges[i,j]
                edge_color = round.(255*collect(cmap.to_rgba(target_edges[i,j]))[1:3], digits=2)
                edge_file[nn,3] = edge_color[1];
                edge_file[nn,4] = edge_color[2];
                edge_file[nn,5] = edge_color[3];
            end
        end
    
        writedlm(datadir(location, name*"edges.csv"), edge_file, ",")
    
        ### Exporting residual edges
        number_edges=count(i->(abs(i)>threshold), residual_edges)÷2
        edge_file_residual = Array{Any,2}(nothing, number_edges +1 ,5)
        header = ["u" "v" "R" "G" "B"]
        edge_file_residual[1,:] = header
    
        nn=1
        for i=1:Nions, j=i+1:Nions
            if abs(residual_edges[i,j]) > threshold
                nn+=1
                edge_file_residual[nn,1]=i+Nions
                edge_file_residual[nn,2]=j+Nions
                edge_color = round.(255*collect(cmap.to_rgba(residual_edges[i,j]))[1:3],digits=2)
                edge_file_residual[nn,3]=edge_color[1];
                edge_file_residual[nn,4]=edge_color[2];
                edge_file_residual[nn,5]=edge_color[3];
            end
        end
        writedlm(datadir(location, name*"edges_residual.csv"), edge_file_residual, ",")
    end


"================="
# Debugging
"================="


gs_debug= []

function g_sum!(G, parms)
    Jₑ = Jexp_noμ(parms)
    #λ, U = eigsystem(parms)
    AA = sum([A̅ₖₗ(i,j) for i=1:Nions for j=1:Nions])
    sum(isnan.(AA)) >= 1 && show(stdout, "text/plain", AA)
    sum(isnan.(AA)) >= 1 && println("Error in gradient")
    ϵⱼ = sum(Jₑ.^2)
    #A̅ₖₗ₂(k,l) = A̅ₖₗ(k,l,U,λ)
    #Do I need to calculate U and λ at each iteration of the gradient? No, see section 2.2.1 of Miles
    for k=1:length(parms)
        G[k] = sum([2*Jₑ[i,j]*A̅ₖₗ(i,j)[k,k] for i=1:Nions for j=1:Nions])
        isnan(G[k]) == true && println("Error in gradient")
        println("Calculating")
    end
    push!(gs_debug, G)
end

function fg_sum!(F,G,parms)
    Jₑ = Jexp_noμ(parms)
    ϵⱼ = sum(Jₑ.^2)
    if G != nothing
        for k=1:length(parms)
            G[k] = sum([2*Jₑ[i,j]*A̅ₖₗ(i,j)[k,k] for i=1:Nions for j=1:Nions])
        end
    end
    if F != nothing
      value = ϵⱼ
      return value
    end
end

xs = []; gs = [];
cb = tr -> begin
            push!(xs, tr[end].metadata["x"])
            push!(gs, tr[end].metadata["g(x)"])
            false
        end



cbg = tr -> begin
            push!(gs, tr[end].metadata["g(x)"])
            false
        end
