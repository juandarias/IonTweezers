using Distributed; addprocs(6)

base_directory = "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Code/Julia/Toffoli/"
framework_directory = base_directory*"framework/";
output_location = base_directory*"tests/Toffoli/Pinning/"
input_location = base_directory*"input/"
push!(LOAD_PATH, framework_directory)

using LinearAlgebra, SharedArrays, Optim, JLD, LineSearches #, Nabla, NLopt
using optimizers, coupling_matrix, crystal_geometry





"===================="
# Global definitions
"===================="
    #
    ⊗ = kron
    ω_trap=[Ωx(3500,1000,3200,2*π*1E7),Ωy(3500,1000,3200,2*π*1E7),Ωz(3200)]/1E6


"===================="
# Coupling matrices
"===================="

    Nions = 7;

    JMM=zeros(Nions,Nions,17); ω_z=zeros(Nions,17); δ_z=zeros(Nions,17); bm_z=zeros(Nions,Nions,17);n=0;
    for omega in 1:0.25:5
        n+=1
        ω_trap_m = omega*ω_trap
        pos_ions = sort(PositionIons(Nions,ω_trap_m*1E6,plot_position=false), dims=2,by=abs);
        Jexp(parms)  = JexpAD(pos_ions, parms[1:Nions], ω_trap_m, parms[Nions+1], kvec=[0.0,0.0,1.0])
        parm_nopin = append!(zeros(Nions), ω_trap_m[3] - 2*pi*0.2);
        JNP, evals, evecs = Jexp(parm_nopin);
        JMM[:,:,n] = JNP./maximum(-JNP);
        ω_z[:,n] = evals[[2,4,7,12,19,20,21]];
        δ_z[:,n] = -ω_z[:,n] .+ (ω_trap_m[3] - 2*pi*0.2);
        bm_z[:,:,n] = evecs[:,[2,4,7,12,19,20,21]][15:21,:];
        #err = norm(JMM3ion/norm(JMM3ion) + JCoMN)
        #push!(errors5, err)
    end


    @save input_location*"Parameters_MM_7ion_A200.jld" ω_z JMM δ_z bm_z


    mm7ion["JMM"][:,:,1]

    JMM[:,:,1]

    mm3ion=load(input_location*"Parameters_MM_3ion_A20.jld");
    mm5ion=load(input_location*"Parameters_MM_5ion_A20.jld");
    mm7ion=load(input_location*"Parameters_MM_7ion_A20.jld");


    function TargetPhases(modes::Array{Float64,2}, Jtarget::Array{Float64,2})
        Nions = size(modes)[2];
        Oj(m,n)= transpose([(transpose(modes[:,j]) ⊗ modes[:,j])[m,n] for j in 1:Nions]);
        AA = vcat([Oj(m,n) for n = 1:Nions-1 for m = n+1:Nions]...);
        Jvector=[Jtarget[m,n] for n = 1:Nions-1 for m = n+1:Nions];
        return pinv(AA)*Jvector
    end

    mm3ion["ω_z"][:,1]

    𝛗3ion = TargetPhases(mm3ion["bm_z"][:,:,1], -0.0355mm3ion["JMM"][:,:,1])
    𝛗MB = [0.0706522,-0.0348152,-0.035837];
    rMB = [0.34358, -7.91309E-8, 1.275E-8, 0, -0.337882, 4.13113E-10, 3.44843E-8, -0.0689791, 0.1723, -0.0324286];

    rMB7 =[0.20909,-7.51219*10^-7,1.26602*10^-7,0,-0.351866,8.26591*10^-7,-0.00888621,0.104058,-0.246807,-0.151416,-0.0904174,0.143513,0.268044,-0.00449836,0.00014595,0.186122, 0.0266433,0.0807128,-0.149106,-0.119494]

    N = 10; offset = 0;T = 4; #parameters of optimization
    μ(i) =  (i + offset)/T;

format_Plot.SquareSmallerPlot()

function PlotSpectraArticle2(phonon_frequencies, beatnote_frequencies, amplitude)
    fig,ax=plt.subplots(figsize=[4.1,2.5]);
    (markerline_u, stemlines_u, baseline_u) = stem(beatnote_frequencies, amplitude, linefmt="C2-", markerfmt="C2o",basefmt="None")
    setp(stemlines_u, linewidth=2)
    tick_params(axis="both", which="major", labelsize=14)
    for ωf in phonon_frequencies
        axvline(ωf,ls="--",c="C3")
    end
    ylabel("\$ \\Omega_\\mu/(2\\pi) \$ (MHz)",fontsize=16)
    xlabel("\$\\mu/(2\\pi)\$ (MHz)",fontsize=16)
    ylim(0)
    #xscale("log")
end


function PlotSpectraBeatNotes(phonon_frequencies, beatnote_frequencies, amplitude)
    fig,ax=plt.subplots(figsize=[4.1,2.5]);
    (markerline_u, stemlines_u, baseline_u) = stem(beatnote_frequencies, amplitude, linefmt="C2-", markerfmt="C2o",basefmt="None")
    setp(stemlines_u, linewidth=2)
    for ωf in phonon_frequencies
        axvline(ωf,ls="--",c="C3")
    end
    ylabel("\$ \\Omega_\\mu/(2\\pi) \$ (MHz)",fontsize=16)
    xlabel("\$\\mu/(2\\pi)\$ (MHz)",fontsize=16)
    ylim(0)
    #xscale("log")
end


    PlotSpectraBeatNotes(mm7ion["ω_z"][:,1]./(2*pi),[μ(n) for n=1:10],abs.(rMB7))

    savefig(output_location*"beatnote_3ion_A20_1.svg")



    @save input_location*"JIsingMB_3ion_A20_1.jld" JMB 𝛗MB rMB

    bm_z1=mm3ion["bm_z"][:,:,1];
    J3ionMB = sum([𝛗3ionMB[j]*kron(bm_z1[:,j]',bm_z1[:,j]) for j = 1:Nions]).*[i==j ? 0 : 1 for i = 1:Nions, j = 1:Nions]

    JMB=J3ionMB/maximum(J3ionMB)
    J3t=mm3ion["JMM"][:,:,1]

    norm(JMB+J3t)/norm(J3t)



    𝛗7ion = TargetPhases(mm7ion["bm_z"][:,:,1], 5*mm7ion["JMM"][:,:,1])
    𝛗7ion = TargetPhases(mm7ion["bm_z"][:,:,1], mm7ion["JMM"][:,:,1])

    JIsing=η0^2*Ωr^2/(4*δf);

    Jvector=round.([mm7ion["JMM"][:,:,1][n,m] for n = 1:7 for m = n+1:7],digits=4)


    @save input_location*"Parameters_MM_7ion_A20.jld" JMM ω_z δ_z bm_z
















    fig,ax1= plt.subplots();
    ax1.scatter(5:25:1000,errors3delta,label=3);ax1.scatter(5:25:1000,errors5delta,label=5);ax1.scatter(5:25:1000,errors7delta,label=7);
    ax1.set_xlabel("\$ \\delta/2\\pi\$ [kHz]");
    ax1.set_ylabel("\$ \\epsilon\$");
    legend()

    fig,ax1= plt.subplots();
    ax1.scatter(1:0.1:5,errors3,label=3);ax1.scatter(1:0.1:5,errors5,label=5);ax1.scatter(1:0.1:5,errors7,label=7);
    ax1.set_xlabel("\$ \\Omega_{z}/2\\pi\$ [MHz]");
    ax1.set_ylabel("\$ \\epsilon\$");
    ylim(0,0.006)
    legend()







    # min_err=SharedArray{Float64}(10);solns=SharedArray{Float64}(Nions+2,10);
    # #for n in 1:10
    # @sync @distributed for n in 1:10
    #     μinitial, μmax, μmin = ω_trap[3]-2*pi*0.02, ω_trap[3]+2*pi*0.2, ω_trap[3]-2*pi*0.2;
    #     Ωinitial,  c1initial = rand(Nions), 1 #frequencies in 2 pi MHz
    #     initial_parms = vcat(Ωinitial, μinitial, c1initial);
    #     J1D(parms) = JexpAD(pos_ions, parms[1:Nions], ω_trap, parms[Nions+1], kvec=[0.0,0.0,1.0])[1]
    #     ϵ_J(parms) = norm((J1D(parms)/norm(J1D(parms))) .* parms[Nions+2] - Array(JCoMN))
    #     ### Seeds
    #     try
    #         res = GDPin(Nions, ϵ_J, initial_parms, 2*pi*0.5; μmin = μmin, μmax = μmax, c1max = 10.0, c1min = 0.1, show_trace=true)
    #         min_err[n] = res.minimum
    #         solns[:,n] = res.minimizer
    #     catch error
    #         println(error)
    #     end
    # end


    min_err_I=SharedArray{Float64}(10);solns_I=SharedArray{Float64}(Nions+2,10);
    #for n in 1:20
    @sync @distributed for n in 1:10
        μinitial, μmax, μmin = ω_trap[3]-2*pi*0.02, ω_trap[3]-2*pi*0.010, ω_trap[3]-2*pi*0.1;
        c1min, c1max, c1initial = -10, 10, 1;
        Ωinitial, Ωmax = rand(Nions), 2*pi*1.0 #frequencies in 2 pi MHz
        initial_parms = vcat(Ωinitial, μinitial, c1initial);
        J1D(parms) = JexpAD(pos_ions, parms[1:Nions], ω_trap, parms[Nions+1], kvec=[0.0,0.0,1.0])[1]
        ϵ_J(parms) = norm((J1D(parms)/norm(J1D(parms))) .* parms[Nions+2] - Array(JCoMN))
        lc = append!(zeros(Nions), [μmin, c1min]);
        uc = append!(Ωmax*ones(Nions), [μmax, c1max]);
        ### Seeds
        try
            inner_optimizer = BFGS(linesearch=LineSearches.BackTracking());
            res = Optim.optimize(ϵ_J, lc, uc, initial_parms, Fminbox(inner_optimizer))
            min_err_I[n] = res.minimum
            solns_I[:,n] = res.minimizer
        catch error
            println(error)
        end
    end



    JexpR(parms)  = JexpAD(pos_ions, parms[1:Nions], ω_trap, parms[Nions+1], kvec=[0.0,0.0,1.0])
    JexpRT(parms)  = JexpAD(pos_ions, parms[1:Nions], ω_trap, parms[Nions+1], kvec=[0.0,0.1,0.0])
    JexpInv(parms)  = JexpAD(pos_ions, parms[1:Nions], ω_trap, μinv, kvec=[0.0,0.0,0.1])[1]
    evalsR(parms)  = JexpAD(pos_ions, parms[1:Nions], ω_trap, parms[Nions+1], kvec=[0.0,0.0,0.1])[2]

    parm_nopin = append!(zeros(Nions), ω_trap[3] - 2*pi*0.01)

    for n in 1:10
        Jij, evals, evecs = JexpR(solns_I[:,n])
        diff = (Jij[1,2]-Jij[1,3])/Jij[1,2]
        δm = -evals[1].+solns_I[:,n][Nions+1]
        println([Jij[1,2], diff, δm*1E6/(2*pi)])
    end

    Jij, evals, evecs= JexpR(solns_I[:,3])
    Jij, evals, evecs= JexpR(parm_nopin);

    evecs[:,1]
    findall(x->x!=0.0, round.([evecs[:,n] ⋅ (append!(zeros(3*Nions-1),1)) for n in 1:3*Nions],digits=4))
    δm/(2*pi)

    evals_z = evals[[1,6,9]]
    evals_z = evals[[1,2,3,8,15]] #5ions
    evals_z = evals[[2,4,7,12,19,20,21]] #7ions
    δm = -evals_z.+solns_I[:,3][Nions+1]
    δm = -evals_z.+parm_nopin[Nions+1]
    evecs_z = evecs[:,[1,6,9]][7:9,:] #3ions
    evecs_z = evecs[:,[1,2,3,8,15]][11:15,:]; #5ions
    evecs_z = evecs[:,[2,4,7,12,19,20,21]][15:21,:] #5ions

    soln = solns_I[:,3];
    soln = parm_nopin;
    freqs=reshape([evals_z..., δm...],Nions,2);
    @save output_location*"Jmatrix_7ion_new.jld" Jij freqs evecs_z soln
    res5=load(output_location*"Jmatrix_5ion_I.jld")

    res5["evecs"][11:15,[1,4,7,14,15]]



    evecs_z





    vals=filter(x->x!=0,[Jij...])

    (minimum(vals)-maximum(vals))/minimum(vals)

    evals[[1,4,7,14,15]]
    evecs[11:15,:][:,[1,4,7,14,15]]

    δm = -evals.+parm7_nopin[Nions+1];
