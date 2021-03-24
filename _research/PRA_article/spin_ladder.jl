using DrWatson, LinearAlgebra, Optim, LineSearches, PyPlot, DelimitedFiles, BSON
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using coupling_matrix, crystal_geometry, plotting_functions



"========================="
# Auxiliary functions
"========================="

    Chop_Sort(x) = sort(round.(x,digits=5), dims=2)
    Chop_Sort2D(A) = round.(A[:, sortperm(A[3,:])],digits=5)
    diagless(A) = A - diagm(diag(A))
    Σ = sum
    ⊗(A,B) = kron(A',B)

"===================================="
#  Parameters and constants
"===================================="

    MHz = 1E6; μm = 1E-6;
    Nions = 12;

    ### Frequencies ###
    ωtrap = 2π*MHz*[0.8, 0.5, 0.2];
    ωtrap = 2π*MHz*[0.6, 0.4, 0.14]; #lowest error

#μnote = 0.8132 #2π 100 kHz red detuned from first phonon mode

"========================="
# Positions and Hessian
"========================="

    pos_ions, fig_crystal = PositionIons(Nions,ωtrap,plot_position=true, tcool=500E-6, cvel=10E-20)
    pos_ions = Chop_Sort2D(pos_ions)

"========================="
# Target and experimental matrices
"========================="

    Jladder = [i==(j+2) ? 0.5*(j!=1)*(i!=Nions) : 0 for i=1:Nions, j=1:Nions] + [i==(j+1) ? -1*(j!=1)*(i!=Nions) : 0 for i=1:Nions, j=1:Nions] + [(j==(i+1) && i in [1,Nions-1]) ? 0.5 : 0 for i=1:Nions, j=1:Nions];
    Jladder = Jladder + Jladder'

    hessY = Hessian(pos_ions, ωtrap; planes=[2]);
    hessZ = Hessian(pos_ions, ωtrap; planes=[3]);
    hessXYZ = Hessian(pos_ions, ωtrap; planes=[1,2,3]);
    hessYZ = hessXYZ[Nions+1:3*Nions,Nions+1:3*Nions]

    Jexp(parms) = Jexp2D(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+1], hessian=hessYZ)[1]

"========================="
# Optimization
"========================="

    # Seed and constrains
    Ωpin(parms) = vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2]), parms[1:Nions÷2], reverse(parms[1:Nions÷2]))

    initial_μ = 5.0; Ωmax = 5.0; μmin = 0.5; μmax = 30; cmin = 1E-2; cmax = 1E4;

    # with proportionality constants
    initial_parms = vcat(0.05*ones(Nions÷2), [initial_μ, 1]);
    lc = append!(zeros(Nions÷2), [μmin, cmin]);
    uc = append!(Ωmax*ones(Nions÷2), [μmax, cmax]);

    # without proportionality constants
    initial_parms = vcat(0.05*ones(Nions÷2), [initial_μ]);
    lc = append!(zeros(Nions÷2), [μmin]);
    uc = append!(Ωmax*ones(Nions÷2), [μmax]);

    # Objective functions
    DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./(abs(coupling_matrix[3,2]))
    DividebyDiag(coupling_matrix::Array{Float64,2}) = coupling_matrix./(sum(diag(coupling_matrix,2))/8)

    ϵ_J1(parms) = norm(DividebyMax(Jexp(parms)) - Jladder)
    ϵ_J2(parms) = 10*norm(normalize(Jexp(parms)) - normalize(Jladder))
    ϵ_J3(parms) = norm(DividebyDiag(Jexp(parms)) - Jladder)
    ϵs_old = (ϵ_J1, ϵ_J2, ϵ_J3)

    ϵ_Frobenius(parms) = norm(Jladder-parms[Nions÷2+2]*Jexp(parms),2) #Frobenius norm
    ϵ_Nuclear(parms) = nucnorm(Jladder-parms[Nions÷2+2]*Jexp(parms)) #Nuclear norm
    ϵ_Spectral(parms) = specnorm(Jladder-parms[Nions÷2+2]*Jexp(parms)) #Nuclear norm
    ϵs = (ϵ_Frobenius, ϵ_Nuclear, ϵ_Spectral)
   
    # Algorithms
    line_search = (HagerZhang(), BackTracking(), BackTracking(order=2));
    algorithm(n) = (LBFGS(linesearch = line_search[n]), BFGS(linesearch = line_search[n]), GradientDescent(linesearch = line_search[n]), ConjugateGradient(linesearch = line_search[n]))


    allresults_LBFGS = [];
    for e in 1:length(ϵs), l in 1:length(line_search), a in 1:4
        try 
            solution = Optim.optimize(ϵs[e], lc, uc, initial_parms, Fminbox(algorithm(l)[a]), Optim.Options(time_limit = 500.0, store_trace=true))
        catch error
            println(error)
        end
        summary_sol = Dict(:method => summary(solution), :objective => string(ϵs[e]), :linesearch => line_search[l], :minimizer => solution.minimizer, :minimum => solution.minimum, :iterations => solution.iterations, :time => solution.time_run)
        push!(allresults_LBFGS, summary_sol)
    end


"========================="
# Plot and export results
"========================="


    for n in 1:9
        note=allresults_LBFGS_old[n][:objective]*"_"*allresults_LBFGS_old[n][:method]*"_"*string(allresults_LBFGS_old[n][:linesearch])[1:1]*"_1.0z\n"*string(round.(solutions_LBGFS_old[n].minimizer,digits=2))*"\n time(s):"*string(round(allresults_LBFGS_old[n][:time],digits=1))
        PlotMatrixArticle(Jsoln(solutions_LBGFS_old[n].minimizer), comment=note);display(gcf())
    end


    for n in 1:24   
        wsave(datadir("Ladder", savename(allresults_YZ[n], "bson")), allresults_YZ[n])
    end

"========================="
# Import results
"========================="

    using DataFrames, BSON

    Jladder = [i==(j+2) ? 0.5*(j!=1)*(i!=Nions) : 0 for i=1:Nions, j=1:Nions] + [i==(j+1) ? -1*(j!=1)*(i!=Nions) : 0 for i=1:Nions, j=1:Nions] + [(j==(i+1) && i in [1,Nions-1]) ? 0.5 : 0 for i=1:Nions, j=1:Nions]
    Jladder = Jladder + Jladder'

    spin_ladder_results = collect_results(datadir("Ladder"))
    hessian_ladder = readdlm(datadir("Ladder","hessianYZ.csv"),',', Float64, '\n')
    DividebyMax(matrix) = matrix./maximum(abs.(matrix))

    names(spin_ladder_results)
    method_error = spin_ladder_results[!,[:method,:minimum]]
    spin_ladder_LBFGS= spin_ladder_results[spin_ladder_results.method .=="Fminbox with L-BFGS",[:method,:objective,:minimizer]]

    spin_ladder_LBFGS[9,:minimizer]
    sol_ladder = spin_ladder_LBFGS[12,:minimizer]

    μsol = sol_ladder[Nions÷2+1]*ωtrap[3]/(2π*MHz)


    Jexp = Jexp2D(pos_ions, Ωpin(sol_ladder), ωtrap, sol_ladder[Nions÷2+1], hessian=hessian_ladder)[1]
    JexpN = Jexp./maximum(abs.(Jexp))
    Ωsol =  Ωpin(sol_ladder)*ωtrap[3]/(2π*MHz)

    ϵ_target, ϵ_residual = CouplingErrors(Jladder,JexpN)

    ExportTikzTweezerGraph(pos_ions_sorted, Ωsol[1:12], "Spin_Ladder_LBFGS", "Article")

    ExportTikzCouplingGraphError(pos_ions, Jladder, abs.(ϵ_target), abs.(ϵ_residual), "Spin_Ladder_LBFGS_abs_error", "Article")

"============================="
# Benchmarking optimizers
"============================="


    allresults_LBFGS_old = [];
    solutions_LBGFS_old = [];
    a=1;
    for e in 1:length(ϵs_old), l in 1:length(line_search)
        try 
            solution = Optim.optimize(ϵs_old[e], lc, uc, initial_parms, Fminbox(algorithm(l)[a]), Optim.Options(time_limit = 250.0, store_trace=true))
        catch error
            println(error)
        end
        summary_sol = Dict(:method => summary(solution), :objective => string(ϵs_old[e]), :linesearch => line_search[l], :minimizer => solution.minimizer, :minimum => solution.minimum, :iterations => solution.iterations, :time => solution.time_run)
        push!(allresults_LBFGS_old, summary_sol)
        push!(solutions_LBGFS_old, solution)
    end

    fig, (ax1, ax2) = plt.subplots(2, 1)
    for n in [1,2,3,4,5,6,8,9]
        note=allresults_LBFGS_old[n][:objective]*"_"*allresults_LBFGS_old[n][:method][14:end]*"_"*string(allresults_LBFGS_old[n][:linesearch])[1:1]
        TracePlotter(solutions_LBGFS_old[n], note)
    end
    ax1.legend()
    display(gcf())





























    # Optimization
    solutionGD_1 = Optim.optimize(ϵ_J1, lc, uc, initial_parms, Fminbox(GradientDescent()), Optim.Options(time_limit = 3.0)) #converges
    solutionGD_2 = Optim.optimize(ϵ_J2, lc, uc, initial_parms, Fminbox(GradientDescent())) #too slow to converge
    solutionLBFGS_1 = Optim.optimize(ϵ_J1, lc, uc, initial_parms, Fminbox(LBFGS())) #converges
    solutionLBFGS_2 = Optim.optimize(ϵ_J2, lc, uc, initial_parms, Fminbox(LBFGS())) #converges
    solutionLBFGS_3 = Optim.optimize(ϵ_J3, lc, uc, initial_parms, Fminbox(LBFGS())) #converges
    solutionLBFGS_1a = Optim.optimize(ϵ_J1, lc, uc, initial_parms, Fminbox(inner_optimizer)) #converges
    solutionLBFGS_2a = Optim.optimize(ϵ_J2, lc, uc, initial_parms, Fminbox(inner_optimizer)) #converges
    solutionLBFGS_3a = Optim.optimize(ϵ_J3, lc, uc, initial_parms, Fminbox(inner_optimizer)) #converges
    solutionBFGS_1 = Optim.optimize(ϵ_J3, lc, uc, initial_parms, Fminbox(inner_optimizer_2)) #converges
    solutionBFGS_2 = Optim.optimize(ϵ_J2, lc, uc, initial_parms, Fminbox(inner_optimizer_2)) #converges
    solutionBFGS_3 = Optim.optimize(ϵ_J1, lc, uc, initial_parms, Fminbox(inner_optimizer_7)) #converges
    solutionGD_3 = Optim.optimize(ϵ_J1, lc, uc, initial_parms, Fminbox(inner_optimizer_3)) #converges, although slower than LBFGS
    solutionGD_4 = Optim.optimize(ϵ_J2, lc, uc, initial_parms, Fminbox(inner_optimizer_3)) #converges, although slower than LBFGS
    solutionGD_5 = Optim.optimize(ϵ_J3, lc, uc, initial_parms, Fminbox(inner_optimizer_3)) #converges, although slower than LBFGS
    solutionCGD_1 = Optim.optimize(ϵ_J1, lc, uc, initial_parms, Fminbox(inner_optimizer_4)) #converges, although slower than LBFGS
    solutionCGD_2 = Optim.optimize(ϵ_J1, lc, uc, initial_parms, Fminbox(inner_optimizer_5)) #converges, although slower than LBFGS
    solutionCGD_3 = Optim.optimize(ϵ_J3, lc, uc, initial_parms, Fminbox(inner_optimizer_4)) #failed non-finite
    solutionCGD_4 = Optim.optimize(ϵ_J2, lc, uc, initial_parms, Fminbox(inner_optimizer_5)) #converges, although slower than LBFGS
    solutionCGD_5 = Optim.optimize(ϵ_J1, lc, uc, initial_parms, Fminbox(inner_optimizer_9)) #converges, although slower than LBFGS
    solutionAGD_1 = Optim.optimize(ϵ_J1, lc, uc, initial_parms, Fminbox(inner_optimizer_6)) #converges, although slower than LBFGS

    gcf()


    Jsoln(solution) = Jexp(solution)./maximum(abs.(Jexp(solution)))

    PlotMatrixArticle2(Jsoln(solutionGD_4), comment="GD_MoreTuente_ϵ2_0.0y_1.0z");gcf()
    PlotMatrixArticle(Float64.(Jladder));gcf()  
    solutionLBFGS_1.minimizer




"============="
# Observations
"============="

#-no solution using only z-modes
