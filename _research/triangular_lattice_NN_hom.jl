using Distributed
addprocs(8)

@everywhere begin
    using Pkg;
    Pkg.activate("./");
    #using DrWatson, BSON, LinearAlgebra, Arpack, DelimitedFiles, SparseArrays, Revise
    using DrWatson, LinearAlgebra, Arpack, SparseArrays, BSON, DelimitedFiles, SharedArrays, Dates
    push!(LOAD_PATH, srcdir());
    push!(LOAD_PATH, scriptsdir());
    include(srcdir()*"/hamiltonians.jl");
    include(srcdir()*"/observables.jl");
    using operators_basis
end

#* Build NN coupling matrix and basis

@everywhere begin
    using models.Kondo, models.general

    d₀ = 1.5;
    depth = 2;
    pos_ions_even = TriangularLattice(d₀, depth);
    NN, iNN = NearestNeighbour(pos_ions_even, 1.6);

    Jhom = Float64.(NN);
    basis19 = generate_basis(19);
    lattice_labels = SublatticesABC()

    JAFM = readdlm(datadir("TriangularLattice", "AFM_Triangular_Coupling_Matrix.dat"));

    U = spdiagm(append!(ones(7),zeros(12))) + spdiagm(1=>append!(zeros(7),ones(11))); U[19,8] = 1.0;
    JAFM_19 = U*JAFM*transpose(U) #hack to rotate spin indices to match SublatticesABC
    JAFM_19_def = copy(JAFM_19)
    JAFM_19_def[1,2] = -1;JAFM_19_def[2,1] = -1

end

#* Calculate ground states
    Base.convert(SparseMatrixCSC, tf::TransverseFieldIsing) = sparse(tf.row_ixs, tf.col_ixs, tf.nz_vals)


    ψ0_hom = SharedArray{ComplexF64}(2^19, 16);
    h = collect(0:0.2:3);
    @sync @distributed for n =1:length(h)
        TFIM = TransverseFieldIsing(Jhom,[h[n]]);
        TFIM_H = convert(SparseMatrixCSC, TFIM);
        λ0, ϕ0 = eigs(TFIM_H, nev=1, which=:SR, v0=ones(ComplexF64, 2^19));
        ψ0_hom[:,n] = ϕ0[:,1];
    end

    
    using BSON: @save, @load

    ψ0 = Matrix(ψ0_hom)
    filename = datadir("TFIM", string(Dates.today())*"_19_ions_hom_ground_states.bson")
    @save filename ψ0

    ψ0_uneven = SharedArray{ComplexF64}(2^19, 16);
    h = collect(0:0.2:3);
    @sync @distributed for n =1:length(h)
        TFIM = TransverseFieldIsing(JAFM_19,[h[n]]);
        TFIM_H = convert(SparseMatrixCSC, TFIM);
        λ0, ϕ0 = eigs(TFIM_H, nev=1, which=:SR, v0=ones(ComplexF64, 2^19));
        ψ0_uneven[:,n] = ϕ0[:,1];
    end

    ψ0 = Matrix(ψ0_uneven)
    filename = datadir("TFIM", string(Dates.today())*"_19_ions_exp_ground_states.bson")
    @save filename ψ0

    ψ0_def = SharedArray{ComplexF64}(2^19, 16);
    h = collect(0:0.2:3);
    @sync @distributed for n =1:length(h)
        TFIM = TransverseFieldIsing(JAFM_19_def,[h[n]]);
        TFIM_H = convert(SparseMatrixCSC, TFIM);
        λ0, ϕ0 = eigs(TFIM_H, nev=1, which=:SR, v0=ones(ComplexF64, 2^19));
        ψ0_def[:,n] = ϕ0[:,1];
    end

    ψ0 = Matrix(ψ0_def);
    filename = datadir("TFIM", string(Dates.today())*"_19_ions_def_ground_states.bson");
    @save filename ψ0

    ##### Low field ground states #####

    ψ0_hom = SharedArray{ComplexF64}(2^19, 20);
    h = 10 .^(range(-2.0, 0, length=20));
    @sync @distributed for n =1:length(h)
        TFIM = TransverseFieldIsing(Jhom,[h[n]]);
        TFIM_H = convert(SparseMatrixCSC, TFIM);
        λ0, ϕ0 = eigs(TFIM_H, nev=1, which=:SR, v0=ones(ComplexF64, 2^19));
        ψ0_hom[:,n] = ϕ0[:,1];
    end

    ψ0 = Matrix(ψ0_hom)
    filename = datadir("TFIM", string(Dates.today())*"_19_ions_hom_LF_ground_states.bson")
    @save filename ψ0


    ψ0_exp = SharedArray{ComplexF64}(2^19, 20);
    h = 10 .^(range(-2.0, 0, length=20));
    @sync @distributed for n =1:20
        TFIM = TransverseFieldIsing(JAFM_19,[h[n]]);
        TFIM_H = convert(SparseMatrixCSC, TFIM);
        λ0, ϕ0 = eigs(TFIM_H, nev=1, which=:SR, v0=ones(ComplexF64, 2^19));
        ψ0_exp[:,n] = ϕ0[:,1];
    end

    ψ0 = Matrix(ψ0_exp)
    filename = datadir("TFIM", string(Dates.today())*"_19_ions_exp_LF_ground_states.bson")
    @save filename ψ0

    


#* On-site magnetization
    filename = datadir("TFIM", "2021-05-01_19_ions_hom_ground_states.bson")
    data = BSON.load(filename)
    ψ0 = data[:ψ0];


    @everywhere function OnsiteMagnetizationZ(state, basis)
        nsites = Int(log(2,length(state)));
        mz_i = SharedArray{ComplexF64}(nsites);
        #mz_i = zeros(nsites);
        @sync @distributed for i=1:nsites
            for (n,bstate) in enumerate(basis)
                mz_i[i] += bstate[i] ? state[n]^2 : -state[n]^2
            end
        end
        return mz_i
    end

    function OnsiteMagnetizationZ(state)
        mz_i = zeros(ComplexF64, 19)
        for i=1:19
            mz_i[i] += (state')*σᶻᵢ(i,19)*state
        end
        return mz_i
    end

    function OnsiteMagnetizationX(state)
        mz_i = zeros(ComplexF64, 19)
        for i=1:19
            mz_i[i] += (state')*σˣᵢ(i,19)*state
        end
        return mz_i
    end
    
    function OnsiteMagnetizationY(state)
        mz_i = zeros(ComplexF64, 19)
        for i=1:19
            mz_i[i] += (state')*σʸᵢ(i,19)*state
        end
        return mz_i
    end

    mx_def = zeros(ComplexF64,19,16);
    my_def = zeros(ComplexF64,19,16);
    mz_hom = zeros(ComplexF64,19,16);
    mz_hom = SharedArray{ComplexF64}(19,16);
    @sync @distributed for i =1:16
        #mx_def[:,i] = OnsiteMagnetizationX(ψ0[:,i])
        #my_def[:,i] = OnsiteMagnetizationY(ψ0[:,i])
        mz_hom[:,i] = OnsiteMagnetizationZ(ψ0[:,1], basis19)
    end
    
    mz_i = zeros(2^19)
    for n=1:2^19
        mz_i[n] = (basis19[n][1] ? 1 : -1)
    end

    sum(mz_i)
    basis19[2][1] + basis19[2][1]

    markers = ["x", "+", "^", "*"]
    plt.subplots()
    #plot(transpose(real.(mx_def)));legend();gcf()
    scatter(h, real.(mx_def)[1,:], label="site 1", marker="x", alpha=0.5);
    scatter(h, real.(mx_def)[6,:], label="site 7", marker="+", alpha=0.5);
    scatter(h, real.(mx_def)[6,:], label="site 8", marker="^", alpha=0.5);
    for n=2:7
        scatter(h, real.(mx_def)[n,:], label="site $n", marker=markers[mod(n,4)+1], alpha=0.5);
    end
    gcf()
    xlabel("\$\\Gamma/J\$")
    ylabel("\$\\langle \\hat{\\sigma}^x_i \\rangle\$")
    legend()
    gcf()
    #ylim(4,7.2)
    savefig(plotsdir("TFIM","onsite_x_19_ions_def_def_d0_d1.png"))


    ##### Low field #####

    mx_exp_LF = zeros(ComplexF64,19,20);
    my_exp_LF = zeros(ComplexF64,19,20);
    mz_exp_LF = zeros(ComplexF64,19,20);
    for i =1:20
        mx_exp_LF[:,i] = OnsiteMagnetizationX(ψ0[:,i])
        my_exp_LF[:,i] = OnsiteMagnetizationY(ψ0[:,i])
        mz_exp_LF[:,i] = OnsiteMagnetizationZ(ψ0[:,i])
    end

    plt.subplots()
    plot(transpose(real.(mx_hom_LF)));
    plot(transpose(real.(mx_exp_LF)));
    gcf()

#* Total magnetization

    Mz0_vs_h = zeros(16);
    TMz0_vs_h = zeros(16);
    TMz0_vs_h_o = zeros(16);
    for n=1:16
        Mz0_vs_h[n] = AverageMagnetizationZ(ψ0[:,1], basis19)
        TMz0_vs_h[n] = TotalMagnetizationZ(ψ0[:,1], basis19)
        TMz0_vs_h_o[n] = TotalMagnetizationZ(ψ0[:,1])
    end


#* NN correlator

    filename = datadir("TFIM", string(Dates.today())*"_19_ions_exp_ground_states.bson")
    data = BSON.load(filename)
    ψ0 = data[:ψ0];


    NNcorr_vs_h_exp = zeros(ComplexF64,19,19,16);
    for n=1:16
        NNcorr_vs_h_exp[:,:,n] = NeighbourCorrelatorZ(ψ0[:,n], NN)
    end
    

    for n=1:16
        f = h[n]
        NN_corr = transpose(U)*real.(NNcorr_vs_h_exp[:,:,n] + transpose(NNcorr_vs_h_exp[:,:,n]))*U
        GraphCoupling(NN_corr, position_ions, plane="YZ", zero_offset=0.05, label="\$\\langle \\hat\\sigma_i^z \\hat\\sigma_j^z\\rangle\$", title_plot="\$h=$f\$");
        savefig(plotsdir("TFIM", string(Dates.today())*"_19ions_NN_exp_HF_"*string(f)*".png"))
    end


#* XY order parameter vs field

    ##### Low-field calculation #####
    Oxy_vs_h_exp_LF = zeros(ComplexF64, 2,20)
    for n=1:20
        groundstate = ψ0[:,n]
        for i=1:2^19
            Oxy_vs_h_exp_LF[:,n] += groundstate[i]^2*ClockOrderParameter(lattice_labels, basis19[i])
        end
    end

    plt.subplots()
    scatter(h,Oxy_vs_h_LF[2,:], marker="^", label="ideal");
    scatter(h,Oxy_vs_h_exp_LF[2,:], marker="x", label="exp");
    xscale("log")
    xlabel("\$\\Gamma/J\$")
    ylabel("\$\\langle \\hat{O}_\\textrm{XY} \\rangle\$")
    legend()
    #ylim(4,7.2)
    gcf()
    savefig(plotsdir("TFIM","Oxy_19_ions_hom_exp_LF.png"))

    ##### High-field calculations #####
    Oxy_vs_h = zeros(ComplexF64, 2,31)
    #Oxy_vs_h = SharedArray{ComplexF64}(2, 31)
    h = collect(0:0.2:3)
    for n =1:length(h)
        TFIM = TransverseFieldIsing(Jhom,[h[n]]);
        TFIM_H = convert(SparseMatrixCSC, TFIM)
        λ0, ϕ0 = eigs(TFIM_H, nev=1, which=:SR, v0=ones(ComplexF64, 2^19));
        groundstate = @view ϕ0[:,1];
        for i=1:2^19
            Oxy_vs_h[:,n] += groundstate[i]^2*ClockOrderParameter(lattice_labels, basis19[i])
        end
    end

    Oxy_vs_h_exp = zeros(ComplexF64, 2,31)
    #Oxy_vs_h_exp = SharedArray{ComplexF64}(2, 31)
    h = collect(0:0.2:3)
    for n =1:length(h)
        TFIM = TransverseFieldIsing(JAFM_19,[h[n]]);
        TFIM_H = convert(SparseMatrixCSC, TFIM)
        λ0, ϕ0 = eigs(TFIM_H, nev=1, which=:SR, v0=ones(ComplexF64, 2^19));
        groundstate = @view ϕ0[:,1];
        for i=1:2^19
            Oxy_vs_h_exp[:,n] += groundstate[i]^2*ClockOrderParameter(lattice_labels, basis19[i])
        end
    end

    Oxy_vs_h_exp_def = zeros(ComplexF64, 2,31)
    #Oxy_vs_h_exp_def = SharedArray{ComplexF64}(2, 31)
    h = collect(0:0.2:3)
    for n =1:length(h)
        TFIM = TransverseFieldIsing(JAFM_19_def,[h[n]]);
        TFIM_H = convert(SparseMatrixCSC, TFIM)
        λ0, ϕ0 = eigs(TFIM_H, nev=1, which=:SR, v0=ones(ComplexF64, 2^19));
        groundstate = @view ϕ0[:,1];
        for i=1:2^19
            Oxy_vs_h_exp_def[:,n] += groundstate[i]^2*ClockOrderParameter(lattice_labels, basis19[i])
        end
    end

    using plotting_functions

    NormalPlot()

    plt.subplots()
    scatter(h,Oxy_vs_h[2,1:16], marker="x", label="ideal");
    scatter(h,Oxy_vs_h_exp[2,1:16], marker="+", label="exp");
    scatter(h,Oxy_vs_h_exp_def[2,1:16], marker="v", label="def");
    #scatter(h,Oxy_vs_h_exp_even[2,1:16], marker="v", label="19 ion even");
    xlabel("\$\\Gamma/J\$")
    ylabel("\$\\langle \\hat{O}_\\textrm{XY} \\rangle\$")
    legend()
    #ylim(4,7.2)
    savefig(plotsdir("TFIM","Oxy_19_ions_deffect.png"))
    gcf()

    GraphCoupling(JAFM, position_ions, plane="YZ", zero_offset=0.05);gcf()
    savefig(plotsdir("TFIM", "19_ions_defect_couplings.png"));gcf()


#* Correlation function ⟨σᶻᵢσᶻⱼ⟩ - ⟨σᶻᵢ⟩⟨σᶻⱼ⟩