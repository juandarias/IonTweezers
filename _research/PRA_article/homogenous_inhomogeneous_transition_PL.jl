using DrWatson, LinearAlgebra, Optim, JLD, BSON, LineSearches, PyPlot, DelimitedFiles
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
include(plotsdir()*"/format_Plot.jl")
using optimizers, coupling_matrix, crystal_geometry, plotting_functions


"================"
# Definitions
"================"

    MHz = 1E6; μm = 1E-6;
    Chop_Sort(x) = sort(round.(x,digits=5), dims=2)
    DividebyMax(coupling_matrix::Array{Float64,2}) = coupling_matrix./maximum(coupling_matrix)

"========================="
# Crystal parameters
"========================="

    Nions = 12;
    z0 = 40*10^-6; d = 2*z0/Nions; d += d/(Nions -1); # Homogenous crystal

    #* Frequencies
    ωtrap = 2π*MHz*[0.6, 0.6, 0.1];


    ### Fitting F = ax + bx^3 ###
    a,b = 1.84*10^-14, -0.000110607 

    ### Frequencies ###
    ωtrap_even = 2*pi*1E6*[0.6, 0.6, 0.2];
    ωtrap_even[3] = abs(√(ee^2/(2π*ϵ0*mYb*d^3)))


"========================="
# Positions and Hessian
"========================="

    pos_ions = PositionIons(Nions, ωtrap, plot_position=false, tcool=500E-6, cvel=10E-20)
    pos_ions = Chop_Sort(pos_ions)
    hess_X = Hessian(pos_ions, ωtrap; planes=[1])

    pos_ions_even = PositionIons(Nions, ωtrap_even, d, [a,b], plot_position=false)
    pos_ions_even = Chop_Sort(pos_ions_even)
    hess_X_even = Hessian(pos_ions_even, ωtrap_even, d, [a,b]; planes=[1])


    function PositionEvolution(ref_positions, χ)
        new_positions = zero(ref_positions)
        d₀ = ref_positions[3,Nions÷2] - ref_positions[3,Nions÷2 + 1]
        dₙ = [ref_positions[3,i]-ref_positions[3,i+1] for i=Nions÷2:11]
        Δₙ = dₙ .- d₀
        dₙ̃ = (1-χ)*Δₙ .+ d₀
        xₙ = [ref_positions[3,Nions÷2 + 1] + sum([dₙ̃[n] for n=1:m]) for m=1:Nions÷2]
        new_positions[3,:] = prepend!(-xₙ, reverse(xₙ))
        return new_positions
    end

    Hessian_X(pos) = Hessian(pos, ωtrap; planes=[1])

"========================="
# Optimization and Benchmarking
"========================="

    function PowerLaw_LBFGS(α, trap_frequency, hessian; order_BT=3)
        #* Seed and constrains
        Ωpin(parms) = vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2]))
        μ_seed = 10.0;
        parms_seed = vcat(0.05*ones(Nions÷2), μ_seed);
        Ωmax = 7.0; 
        μmin = 0.1; 
        μmax = 14.0;
        lc = append!(zeros(Nions÷2), μmin);
        uc = append!(Ωmax*ones(Nions÷2), μmax);
        
        #* Target and experimental matrix
        JPL = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]
        Jexp(parms) = Jexp1D(pos_ions, Ωpin(parms), trap_frequency, parms[Nions÷2+1], planes=[1], hessian=hessian)[1];

        #* Objective function
        DividebyDiag(coupling_matrix::Array{Float64,2}) = coupling_matrix./(sum(diag(coupling_matrix,1))/11)
        ϵ_J(parms) = norm(DividebyDiag(Jexp(parms)) - JPL)

        #* Solve
        #solution = Optim.optimize(ϵ_J, lc, uc, parms_seed, Fminbox(LBFGS(linesearch = HagerZhang())), Optim.Options(time_limit = 500.0, store_trace=false))
        solution = Optim.optimize(ϵ_J, lc, uc, parms_seed, Fminbox(LBFGS(linesearch = BackTracking(order=order_BT))), Optim.Options(time_limit = 500.0))
        #solution = Optim.optimize(ϵ_J, lc, uc, parms_seed_HZ, Fminbox(LBFGS(linesearch = BackTracking(order=order_BT))), Optim.Options(time_limit = 500.0, store_trace=true))

        return solution, Jexp(solution.minimizer)
    end

    function PowerLaw_CG(α, trap_frequency, hessian; order_BT=3)
        #* Seed and constrains
        Ωpin(parms) = vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2]))
        μ_seed = 10;
        parms_seed = vcat(0.05*ones(Nions÷2), μ_seed);
        Ωmax = 7.0; 
        μmin = 0.5; 
        μmax = 14;
        lc = append!(zeros(Nions÷2), μmin);
        uc = append!(Ωmax*ones(Nions÷2), μmax);
        
        #* Target and experimental matrix
        JPL = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]
        Jexp(parms) = Jexp1D(pos_ions, Ωpin(parms), trap_frequency, parms[Nions÷2+1], planes=[1], hessian=hessian)[1];

        #* Objective function
        DividebyDiag(coupling_matrix::Array{Float64,2}) = coupling_matrix./(sum(diag(coupling_matrix,1))/11)
        ϵ_J(parms) = norm(DividebyDiag(Jexp(parms)) - JPL)

        #* Solve
        #solution = Optim.optimize(ϵ_J, lc, uc, parms_seed, Fminbox(LBFGS(linesearch = HagerZhang())), Optim.Options(time_limit = 500.0, store_trace=false))
        solution = Optim.optimize(ϵ_J, lc, uc, parms_seed, Fminbox(ConjugateGradient(linesearch = BackTracking(order=order_BT))), Optim.Options(time_limit = 500.0, store_trace=true))

        return solution, Jexp(solution.minimizer)
    end
     

    #* Optimization
    collection_matrices_LBFGS = [];
    collection_summary_LBFGS = [];
    collection_solutions_LBFGS = [];

    errors = [];

    χs = collect(0:0.1:1)
    for αₜ=0.5:0.5:4
        coupling_matrices = zeros(12,12,11);
        powerlaw_LBFGS_summmary = [];
        powerlaw_LBFGS_solution = [];
        for n=1:11
            χ = χs[n]
            hessχ = Hessian_X(PositionEvolution(pos_ions, χ));
            try 
                solution, Jexp = PowerLaw_LBFGS(αₜ, hessχ, order_BT=2)
                summary_sol = SummarySolution(solution, Jexp, "PL_α="*string(αₜ), "||Jₜ-Jₑ÷ave(diag(Jₑ,1))||","LBGFS_O2")
                push!(powerlaw_LBFGS_summmary, summary_sol)
                push!(powerlaw_LBFGS_solution, solution)
                coupling_matrices[:,:,n] = Jexp
            catch error
                println(error)
                push!(errors,[αₜ, χ])
            end
        end
        push!(collection_matrices_LBFGS, coupling_matrices)
        push!(collection_solutions_LBFGS, powerlaw_LBFGS_solution)
        push!(collection_summary_LBFGS, powerlaw_LBFGS_summmary)
    end

    #* Only uneven case

    Jexps_LBFGS = zeros(15,12,12);
    solutions_LBFGS = [];
    n=0;
    for αₜ=0.5:0.25:4
        n+=1
        solution, Jexps_LBFGS[n,:,:] = PowerLaw_LBFGS(αₜ, ωtrap, hess_X, order_BT=2)
        push!(solutions_LBFGS, solution)
    end
    
    Jexps_CG = zeros(15,12,12);
    solutions_CG = [];
    n=0;
    for αₜ=0.5:0.25:4
        n+=1
        solution, Jexps_CG[n,:,:] = PowerLaw_CG(αₜ, ωtrap, hess_X, order_BT=2)
        push!(solutions_CG, solution)
    end


    #* Even case

    Jexps_LBFGS_even = zeros(15,12,12);
    solutions_LBFGS_even = [];
    n=0;
    for αₜ=0.5:0.25:4
        n+=1
        solution, Jexps_LBFGS_even[n,:,:] = PowerLaw_LBFGS(αₜ, ωtrap_even, hess_X_even, order_BT=2)
        push!(solutions_LBFGS_even, solution)
    end

    Jexps_CG_even = zeros(15,12,12);
    solutions_CG_even = [];
    n=0;
    for αₜ=0.5:0.25:4
        n+=1
        solution, Jexps_CG_even[n,:,:] = PowerLaw_CG(αₜ, ωtrap_even, hess_X_even, order_BT=2)
        push!(solutions_CG_even, solution)
    end

    #* Only beatnote

    function ErrorBeatnote(α,μ)
        JPL = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]
        #* experimental matrix
        Jexpnopin = Jexp1D(pos_ions, zeros(12), ωtrap, μ, planes=[1], hessian=hess_X)[1];

        #* Objective function
        ϵ_J = opnorm(Jexpnopin./maximum(abs.(Jexpnopin)) - JPL)/opnorm(JPL)
        return ϵ_J
    end


    errors_μ = zeros(15,10025);
    μs = collect(0.51:0.02:201)
    n=0;
    for αₜ=0.5:0.25:4
        n+=1
        for k =1:10025
            ϵ =  ErrorBeatnote(αₜ,μs[k])
            errors_μ[n,k] = ϵ
        end
    end
    

"==============="
# Error analysis
"==============="

    Jsoln(solution) = Jexp(solution)./maximum(abs.(Jexp(solution)))
    JPL(α) = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]


    function ErrorPowerLaw(coupling_matrices, αs)
        JPL(α) = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]
        ϵ_J(matrix, α) = opnorm(matrix./maximum(abs.(matrix)) - JPL(α))/opnorm(JPL(α))
        error_PL = zeros(15)
        for k=1:length(αs)
            error_PL[k] = ϵ_J(coupling_matrices[k,:,:], αs[k])
        end
        return error_PL
    end
    
    αs = collect(0.5:0.25:4)
    ϵ_even_LBFGS =  ErrorPowerLaw(Jexps_LBFGS_even, αs)
    ϵ_even_CG =  ErrorPowerLaw(Jexps_CG_even, αs)

    ϵ_nopin_res = zeros(15,2);
    ϵ_nopin_res[:,1] = αs
    ϵ_nopin_res[:,2] = ϵ_nopin

    writedlm(datadir("PowerLaw_NH")*"/error_power_law_uneven_nopin.txt", ϵ_nopin_res)



    format_Plot.NormalPlot()
    markers = ["v", "+", "x", "o", "*", "^"]
    prepend!(markers, markers)
    fig, ax = plt.subplots()
    scatter(αs[1:15],ϵ_uneven_LBFGS[1:15], s=20, marker=markers[1], label="\$uneven LBFGS\$");
    #scatter(αs[1:15],ϵ_uneven_LBFGS_HZ[1:15], s=20, marker=markers[2], label="\$uneven LBFGS_HZ\$");
    #scatter(αs[1:15],ϵ_uneven_CG[1:15], s=20, marker=markers[3], label="\$uneven CG\$");
    scatter(αs[1:15],ϵ_even_LBFGS[1:15], s=20, marker=markers[4], label="\$even LBFGS\$");
    #scatter(αs[1:15],ϵ_even_LBFGS_HZ[1:15], s=20, marker=markers[5], label="\$even LBFGS_HZ\$");
    #scatter(αs[1:15],ϵ_even_CG[1:15], s=20, marker=markers[6], label="\$even CG\$");
    scatter(αs[1:15],ϵ_nopin[1:15], s=20, marker=markers[6], label="\$no pin\$");
    legend()
    #ylim(0,0.15)
    gcf()


    fig = plt.figure()
    gs = fig.add_gridspec(4,2, hspace=0.07)
    axxs = gs.subplots(sharex=true)
    k=0
    for n=1:4, m=1:2
        k+=1
        min_lim = 0.95*minimum(ϵ_vs_α_vs_χ[k,:][10:11])
        max_lim = 1.05*maximum(ϵ_vs_α_vs_χ[k,:][1:2])
        axxs[n,m].scatter(0:0.1:1,ϵ_vs_α_vs_χ[k,:], label="\$\\alpha=\$"*string(αs[k]), s=10);axxs[n,m].set_ylim(min_lim, max_lim); axxs[n,m].legend();
    end

    ϵ_vs_α_vs_χ[5,:]
    axxs[4,1].set_xlabel("\$\\chi\$");
    axxs[4,2].set_xlabel("\$\\chi\$");
    gcf()
    axxs[1,1].set_ylim(0.05,0.095)
    axxs[2,2].set_ylim(0.037,0.0385)
    axxs[1,2].set_ylim(0.07,0.096)
    #axxs[3,1].set_ylim(0.085,0.1)
    axxs[3,1].set_ylim(0.012,0.017)
    #axxs[4,2].set_ylim(0.016,0.022)

    gcf()
    savefig(plotsdir("HomtoInhom","LBFGS_BTO2_Diag_Div.svg"));gcf()


"============"
# Solutions
"============"

    solutions_LBFGS = zeros(11,7,8)
    for k=1:8, m=1:11
        solutions_LBFGS[m,:,k] = collection_solutions_LBFGS[k][m].minimizer
    end

    k=7
    fig = plt.figure()
    gs = fig.add_gridspec(11,1, hspace=0.7)
    axs = gs.subplots(sharex=true)
    for n=1:11
        axs[n].bar(1:6,solutions[n,1:6,k], label="χ="*string(χs[n])*", α="*string(αs[k]));
        plt.title("α="*string(αs[k]))
    end

    gcf()

    markers = ["v", "+", "x", "o", "*", "^"]
    prepend!(markers, markers)

    fig, ax = plt.subplots()
    for k=1:8
        scatter(0:0.1:1,solutions_CG[:,7,k],label="\$\\alpha=\$"*string(αs[k]),s=75,marker=markers[k],alpha=0.75)
    end
    xlabel("\$\\chi\$")
    ylabel("\$\\mu\$")
    legend()
    savefig(plotsdir("HomtoInhom","beatnote_CG.svg"))
    gcf()

    for k=1:8
        fig, axs = plt.subplots()
        for n=1:11
            axs.bar(collect(1:6).+0.05*k,solutions_CG[n,1:6,k], 0.025, label="χ="*string(χs[n])*", α="*string(αs[k]));
        end
        #plt.title("α="*string(αs[k]))
        plt.title("χ="*string(χs[n]))
        plt.legend()
        #savefig(plotsdir("HomtoInhom","CG"*"α="*string(αs[k])*".png"))
        savefig(plotsdir("HomtoInhom","CG"*"χ="*string(χs[n])*".png"))
        gcf()
    end


    k=0
    fig = plt.figure(figsize=(16,12))
    gs = fig.add_gridspec(4,2, hspace=0.05)
    axs = gs.subplots(sharex=true)

    for q=1:4, r=1:2
        k+=1
        for n=1:11
            axs[q,r].scatter(collect(1:6),solutions_CG[n,1:6,k], label="\$\\chi=\$"*string(χs[n]), s=8);
            #axs[q,r].annotate("\$\\alpha=\$"*string(αs[k]),(0,0))
            axs[q,r].set_ylabel("\$\\Omega_{\\alpha="*string(αs[k])*"}\$");
            if k==8
                handles, labels = axs[q,r].get_legend_handles_labels()
                fig.legend(handles, labels, loc=(0.1, 0), ncol=6)
            end
        end
    end
    #axs[1,1].set_ylim(0,10)
    axs[4,1].set_xlabel("Ion")
    axs[4,2].set_xlabel("Ion")
    savefig(plotsdir("HomtoInhom","solution_CG_PL.png"))
    gcf()



    fig = plt.figure(figsize=(16,12))
    gs = fig.add_gridspec(2,2, hspace=0.05)
    axs = gs.subplots(sharex=true)
    k=8
    for n=1:11
        axs[2,2].scatter(1:6,solutions_A[n,1:6,k], label="\$\\chi=\$"*string(χs[n]),marker=markers[n]);
        axs[2,2].set_ylabel("\$\\Omega_{\\alpha="*string(αs[k])*"}\$");
    end
    handles, labels = axs[2,2].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.1, 0), ncol=6)
    axs[1,1].set_ylim(3.5,10)
    axs[1,2].set_ylim(9.6,9.9)
    axs[2,1].set_ylim(9.5,9.9)
    axs[2,2].set_ylim(9,10.5)
    axs[2,1].set_xlabel("Ion")
    axs[2,2].set_xlabel("Ion")
    savefig(plotsdir("HomtoInhom","solution_alpha_LBFGS_2.5-4.svg"))

    gcf()

    format_Plot.NormalPlot()

    fig, ax =plt.subplots()
    for n=1:2:15
        scatter(1:6,solutions_CG_even[n].minimizer[1:6], s=50, label="\$\\alpha=\$"*string(αs[n]), marker=markers[(n+1)÷2], alpha=0.75);
    end
    legend(loc=(1.05, 0))
    xlabel("Ion")
    ylabel("\$ \\Omega_p/\\omega_z\$")
    savefig(plotsdir("PowerLaw","solution_CG.svg"))
    gcf()


"============="
# Eigensystem
"============="

    Ωpin(parms) = vcat(parms[1:Nions÷2], reverse(parms[1:Nions÷2]))
    λp(parms, hess) = Jexp1D(pos_ions, Ωpin(parms), ωtrap, parms[Nions÷2+1], planes=[1], hessian=hess)[2];

    λs_A = zeros(13,11,8);

    for k=1:8, n=1:11
        χ = χs[n]
        hessχ = Hessian_X(PositionEvolution(pos_ions, χ));
        λs_A[1:12,n,k] = sqrt.(λp(solutions_A[n,:,k], hessχ))
        λs_A[13,n,k] = solutions_A[n,7,k]
    end

    

"============"
# Export data
"============"
    
    for n=1:15
        sol = solutions_CG_even[n]
        Jexp = Jexps_CG_even[n,:,:]
        Ωsol = Ωpin(sol.minimizer)
        μsol = sol.minimizer[end]
        dict_PL = SummaryOptimization(ωtrap_even, pos_ions_even, hess_X_even, "Power_Law_even α="*string(αs[n]), "||Jₑ/mean(diag(Jₑ,1))-Jₜ||", sol, Ωsol, μsol, Jexp, note=note_PL_even)
        wsave(datadir("PowerLaw", string(Dates.today())*"_PowerLaw_even_CG_alpha"*string(αs[n])*".bson"), dict_PL)
    end

    note_PL_even = "12 ions, radial modes, numerical gradient, Ωmax = 2 MHz, μmax = 4 MHz"
    
    writedlm(datadir("PowerLaw")*"/error_power_law_homogenous.txt", ϵ_vs_α_LBFGS_even)


    Ωα1 = PowerLaw_uneven_α1[:pinning_frequency]
    μα1 = PowerLaw_uneven_α1[:beatnote]
    hessianα1 = PowerLaw_uneven_α1[:hessian]
    ωtrap = 2π*PowerLaw_uneven_α1["trap_frequency [MHZ]"]*MHz

    JNHα1 = Jexp1D(pos_ions, Ωα1, ωtrap, μα1, planes=[1], hessian=hessianα1)[1]
    
    nJNHα1=JNHα1/JNHα1[1,12]
    
    JNHα1_log = replace!(log.(nJNHα1), -Inf => 0.0)

    ExportTikzMatrix(round.(JNHα1_log, digits=2), "PowerLaw_uneven_alpha=1", datadir("Article"))

"============"
# Import data
"============"

    
    PowerLaw_uneven_α1 = BSON.load(datadir("PowerLaw_NH","2021-03-02_PowerLaw_Uneven_alpha1.0.bson"))
    PowerLaw_even_α1 = BSON.load(datadir("PowerLaw","2021-03-02_PowerLaw_even_alpha1.0.bson"))
    PowerLaw_uneven_α35 = BSON.load(datadir("PowerLaw_NH","2021-03-02_PowerLaw_Uneven_alpha3.5.bson"))
    PowerLaw_even_α35 = BSON.load(datadir("PowerLaw","2021-03-02_PowerLaw_even_alpha3.5.bson"))

    ω_trap_even = PowerLaw_even_α35["trap_frequency [MHZ]"]
    ω_trap_uneven = PowerLaw_uneven_α35["trap_frequency [MHZ]"]
    Ωsol_even_35 = PowerLaw_even_α35[:pinning_frequency]*ω_trap_even[3]
    Ωsol_uneven_35 = PowerLaw_uneven_α35[:pinning_frequency]*ω_trap_uneven[3]
    PowerLaw_even_α35[:beatnote]*ω_trap_even[3]
    PowerLaw_uneven_α35[:beatnote]*ω_trap_uneven[3]

    Ωsol_even_1 = PowerLaw_even_α1[:pinning_frequency]*ω_trap_even[3]
    Ωsol_uneven_1 = PowerLaw_uneven_α1[:pinning_frequency]*ω_trap_uneven[3]
    PowerLaw_even_α1[:beatnote]*ω_trap_even[3]
    PowerLaw_uneven_α1[:beatnote]*ω_trap_uneven[3]
    


    solutions_PL_alpha_35 = zeros(13,2)
    
    solutions_PL_alpha_35[1:12,1] = Ωpin(solutions_LBFGS_even[13].minimizer*ωtrap_even[3]/(2π*MHz))
    solutions_PL_alpha_35[1:12,2] = Ωpin(solutions_LBFGS[13].minimizer*ωtrap[3]/(2π*MHz))
    solutions_PL_alpha_35[13,1] = solutions_LBFGS_even[13].minimizer[end]*ωtrap_even[3]/(2π*MHz)
    solutions_PL_alpha_35[13,2] = solutions_LBFGS[13].minimizer[end]*ωtrap[3]/(2π*MHz)

    header = "#Even LBFGS: Ωpin, μ; Uneven LBFGS: Ωpin, μ\n"

    open(datadir("Article","solutions_PL_alpha_35.dat"), "w") do io
        write(io, header)
        writedlm(io, solutions_PL_alpha_35)
    end;
