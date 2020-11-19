module plotting_functions
    #__precompile__(false)

    push!(LOAD_PATH, "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Results/Scripts")
    #using format_Plot

    using LinearAlgebra, PyPlot, SparseArrays, ExponentialUtilities, PyCall

    export Crystal2D, PlotMatrix, GraphCoupling, PlotSpectra, PlotMatrixArticle, PlotUnitary, TweezerStrength, PhaseSpacePlots, Ψ_x, BasisProjection, PhaseSpacePlotsFast, BlochSphere

    function PlotMatrix(interaction_matrix::Array, target_matrix::Array, solution::Vector)
        fig, ax = plt.subplots()
        Nions = size(interaction_matrix)[1]
        mat = ax.matshow(interaction_matrix, cmap="seismic")
        cbar = fig.colorbar(mat)
        Ωmin = string(round.(solution[1:Nions]; digits=2))
        ϵ_JJ = norm(interaction_matrix-target_matrix)/norm(target_matrix)
        text(-1,-1.25,"\$ \\Omega^2 \$ [MHz]="*Ωmin)
        text(-1,-1.5,"\$ \\mu \$ [MHz]="*string(round(solution[Nions+1], digits=4)))
        text(-0.5,-1.5, "\$ \\epsilon \$ ="*string(round(ϵ_JJ;digits=3)))
        lim_color = maximum(abs.(interaction_matrix))
        mat.set_clim(vmin=-lim_color,vmax=lim_color)
        ax.set_xticklabels(collect(0:1:7))
        ax.set_yticklabels(collect(0:1:7))
        cbar.set_label("\$ J_{i,j} \$[Hz]")
        xlabel("Ion i")
        ylabel("Ion j")
        ax.xaxis.set_label_position("top")
    end

    function PlotMatrixArticle(interaction_matrix::Array)
        fig, ax = plt.subplots()
        Nions = size(interaction_matrix)[1]
        mat = ax.matshow(interaction_matrix, cmap="seismic")
        cbar = fig.colorbar(mat)
        lim_color = maximum(abs.(interaction_matrix))
        mat.set_clim(vmin=-lim_color,vmax=lim_color)
        ax.set_xticklabels(collect(0:1:7))
        ax.set_yticklabels(collect(0:1:7))
        cbar.set_label("\$ J_{i,j}/(2\\pi)\\, \$(Hz)")
        xlabel("Ion i")
        ylabel("Ion j")
        #ax.xaxis.set_label_position("top")
        ax.xaxis.tick_bottom()
    end

    function PlotUnitary(unitary_matrix::Array; labels=nothing, save_results=false, location=nothing)
        fig, ax = plt.subplots()
        dims = size(unitary_matrix)[1]; label_0 = copy(labels);
        #labels != 0 && prepend!(label_0, ["0"]);
        mat = ax.matshow(real(unitary_matrix), cmap="RdBu")
        cbar = fig.colorbar(mat,ticks=(-1, 0, 1))
        #lim_color = maximum(unitary_matrix))
        mat.set_clim(vmin=-1,vmax=1)
        ax.set_xticks(0:1:dims-1);ax.set_yticks(0:1:dims-1)
        labels != 0 ? (ax.set_xticklabels(label_0, minor=true, rotation=90, fontsize=10);ax.set_xticklabels(label_0, rotation=90, fontsize=10)) : (ax.set_xticklabels(collect(0:1:dims)); xlabel("i");ylabel("j");)

        labels != 0 ? (ax.set_yticklabels(label_0, minor=true, fontsize=10); ax.set_yticklabels(label_0,fontsize=10)) : ax.set_yticklabels(collect(0:1:dims))
        #cbar.set_label("\$ \\textrm{Re U}(\\tau_g)\$")
        cbar.set_label("\$ \\textrm{Re} (\\hat U_{i\\textrm{Tof}})\$",fontsize=12, labelpad=-1)
        cbar.ax.tick_params(labelsize=10)
        ax.xaxis.tick_bottom()
        save_results == true && savefig(location*"Re.svg", bbox_inches="tight")
        #ax.xaxis.set_label_position("top")
        fig, ax = plt.subplots()
        mat = ax.matshow(imag(unitary_matrix), cmap="RdBu")
        cbar = fig.colorbar(mat,ticks=(-1, 0, 1))
        #lim_color = maximum(abs.(unitary_matrix))
        mat.set_clim(vmin=-1,vmax=1)
        ax.set_xticks(0:1:dims-1);ax.set_yticks(0:1:dims-1)
        labels != 0 ? ax.set_xticklabels(label_0, rotation=90, fontsize=10) : (ax.set_xticklabels(collect(0:1:dims)); xlabel("i");ylabel("j");)
        labels != 0 ? ax.set_yticklabels(label_0, fontsize=10) : ax.set_yticklabels(collect(0:1:dims))
        #cbar.set_label("\$ \\textrm{Im U}(\\tau_g)\$")
        cbar.set_label("\$ \\textrm{Im} (\\hat U_{i\\textrm{Tof}})\$",fontsize=12, labelpad=-1)
        cbar.ax.tick_params(labelsize=10)
        ax.xaxis.tick_bottom()
        save_results == true && savefig(location*"Im.svg", bbox_inches="tight")
        #ax.xaxis.set_label_position("top");ax.set_zlim([0,1])
    end


    function GraphCoupling(interaction_matrix::Array, ion_positions::Array)
        figure()
        Nions = length(ion_positions)
        ### Position ions and edges
        posx = [ion_positions[i][1] for i in 1:Nions]
        posz = [ion_positions[i][3] for i in 1:Nions]
        xquiver = vcat([[posx[j] for i in j+1:Nions] for j in 1:Nions]...)
        zquiver = vcat([[posz[j] for i in j+1:Nions] for j in 1:Nions]...)
        uquiver = vcat([[-posx[j] + posx[i] for i in j+1:Nions] for j in 1:Nions]...)
        vquiver = vcat([[-posz[j] + posz[i] for i in j+1:Nions] for j in 1:Nions]...)
        dist = [norm([uquiver[i]-xquiver[i],vquiver[i]-zquiver[i]]) for i in 1:length(xquiver)]
        nnneighbour = [dist[i] > 4 for i in 1:length(dist)] #Filters next next NN
        deleteat!(xquiver,nnneighbour)
        deleteat!(zquiver,nnneighbour)
        deleteat!(uquiver,nnneighbour)
        deleteat!(vquiver,nnneighbour)

        ### Colors of coupling edges
        color_edge = vcat([[interaction_matrix[j,i] for i in j+1:Nions] for j in 1:Nions]...)
        deleteat!(color_edge,nnneighbour)

        ### Creates figure
        Q= plt.quiver(xquiver,zquiver,uquiver,vquiver, color_edge, angles="xy", scale_units="xy", scale=1, headaxislength=0, headlength=0, headwidth = 1, cmap="seismic")
        Q.set_clim(vmin=-1,vmax=1)
        plt.colorbar(label="\$\\tildeJ_{i,j}\$")
        plt.scatter(posx,posz,s=200, c="black")
    end

    function Crystal2D(ion_positions::Array)
        Nions = length(ion_positions);
        posx = [ion_positions[i][1] for i in 1:Nions]
        posz = [ion_positions[i][3] for i in 1:Nions]
        scatter(posx,posz,color="blue")
        for i in 1:Nions
            annotate(string(i),(posx[i],posz[i]))
        end
    end

    function TweezerStrength(ion_positions::Array, tweezer_freq::Array)
        Nions = length(ion_positions);
        posx = [ion_positions[i][1] for i in 1:Nions]
        posz = [ion_positions[i][3] for i in 1:Nions]
        figure()
        scatter(posx,posz,c=abs.(tweezer_freq),s=200,cmap="Reds")
        colorbar(label="\$ 2\\pi \\Omega_p\$[MHz]")
        for i in 1:Nions
            annotate(string(i),(posx[i]+0.1,posz[i]+0.1))
        end
    end

    function PlotSpectra(phonon_frequencies::Array, beat_note::Float64; unpinned_frequencies=nothing)
        figure()
        modes = length(phonon_frequencies)
        modes_plane = Int(modes*2/3);
        unpinned_frequencies!=nothing && ((markerline_u, stemlines_u, baseline_u) = stem(unpinned_frequencies,collect(1:1:modes), linefmt="C7:", markerfmt="C7h",basefmt="None"))
        markerline, stemlines, baseline = stem(phonon_frequencies[1:modes_plane],collect(1:1:modes_plane), linefmt="b-", markerfmt="bh",basefmt="None")
        markerline_t, stemlines_t, baseline_t = stem(phonon_frequencies[modes_plane+1:modes],collect(modes_plane+1:1:modes), linefmt="g-", markerfmt="gh",basefmt="None")
        stem([beat_note],[22],markerfmt="C3o",linefmt="red")
        plt.setp(stemlines, linewidth=1)
        plt.setp(stemlines_t, linewidth=1)
        unpinned_frequencies!=nothing && plt.setp(stemlines_u, linewidth=0.5)
        xlabel("f [MHz]")
        ylabel("Mode")
        #xscale("log")
    end

    function PlotSpectraArticle(phonon_frequencies::Array, beat_note::Float64; unpinned_frequencies=nothing)
        figure()
        modes = length(phonon_frequencies)
        modes_plane = Int(modes*2/3);
        unpinned_frequencies!=nothing && ((markerline_u, stemlines_u, baseline_u) = stem(unpinned_frequencies,collect(1:1:modes), linefmt="C7:", markerfmt="C7h",basefmt="None"))
        markerline, stemlines, baseline = stem(phonon_frequencies[1:modes_plane],collect(1:1:modes_plane), linefmt="b-", markerfmt="bh",basefmt="None")
        markerline_t, stemlines_t, baseline_t = stem(phonon_frequencies[modes_plane+1:modes],collect(modes_plane+1:1:modes), linefmt="g-", markerfmt="gh",basefmt="None")
        stem([beat_note],[22],markerfmt="C3o",linefmt="red")
        plt.setp(stemlines, linewidth=1)
        plt.setp(stemlines_t, linewidth=1)
        unpinned_frequencies!=nothing && plt.setp(stemlines_u, linewidth=0.5)
        xlabel("\$ \\omega_m/(2\\pi) \$ [MHz]")
        ylabel("Mode")
        #xscale("log")
    end

    function PhaseSpacePlotsE(psi_t::Array{Complex{Float64},2}, creation_op::SparseMatrixCSC{Complex{Float64},Int64}, annihilation_op::SparseMatrixCSC{Complex{Float64},Int64};colorline="r",k=1)
        steps = size(psi_t)[2];
        x_t = [real(k*adjoint(psi_t[:,t])*(creation_op + annihilation_op)*psi_t[:,t]) for t in 1:steps]
        p_t = [real((0.5*im/k)*adjoint(psi_t[:,t])*(creation_op - annihilation_op)*psi_t[:,t]) for t in 1:steps]
        #fig, ax = plt.subplots()
        #ps = ax.scatter(x_t,p_t,c=collect(0:1/(steps-1):1),cmap="Reds", s=0.1)
        ps = ax.plot(x_t,p_t,linewidth=1.0,c=colorline)
        #ps = ax.plot(x_t,p_t,linewidth=1.0)
        #ps.set_clim(vmin=0,vmax=1)
        ax.set_aspect(0.25)
        #ax.set_xlim(minimum(x_t),maximum(x_t))
        #ax.set_ylim(minimum(p_t),maximum(p_t))
        xlabel("\$ \\langle x \\rangle \$"); ylabel("\$ \\langle p \\rangle /\\hbar\$")
    end


    function PhaseSpacePlotsFastU(psi_0::Array{Complex{Float64},1}, hamiltonian::T, creation_op::T, annihilation_op::T, gate_time::Float64, steps::Int; k::Float64 = 1.0, line_plot::Bool=false, start_time::Float64=0.0, units::Bool=false, shift_center::Bool=false, colormap="Reds") where T<:SparseArrays.SparseMatrixCSC{Complex{Float64},Int64}
        psi_t = expv_timestep(collect(start_time:gate_time/steps:gate_time),-im*hamiltonian,psi_0);
        x_t = [real(k*adjoint(psi_t[:,t])*(creation_op + annihilation_op)*psi_t[:,t]) for t in 1:size(psi_t)[2]]
        p_t =[real((0.5*im/k)*adjoint(psi_t[:,t])*(creation_op - annihilation_op)*psi_t[:,t]) for t in 1:size(psi_t)[2]]
        shift_center != false && (p_t = p_t .- p_t[1])
        #fig, ax = plt.subplots() #comment out when plotting multiple plots simultaneously
        line_plot == true  && (ps = ax.plot(x_t,p_t,linewidth=1.0))
        line_plot == false && (ps = ax.scatter(x_t,p_t,c=collect(0:1/(size(psi_t)[2]-1):1),cmap=colormap,s=0.05); ps.set_clim(vmin=0,vmax=1))
        #ax.set_xlim(minimum(x_t),maximum(x_t))
        #ax.set_xlim(-0.02,0.02)
        #ax.set_ylim(minimum(p_t),maximum(p_t))
        ax.set_aspect(0.25)
        units == false && (ax.set_xticks([],[]);ax.set_yticks([],[]));
        ax.spines["right"].set_visible(false);ax.spines["top"].set_visible(false)
        #ax.set_xlabel("\$ \\langle x \\rangle \$");
        ax.set_ylabel("\$ \\langle p \\rangle / \\hbar \$")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        return x_t, p_t
        #cbar=fig.colorbar(ps,ticks=(0,1)); cbar.ax.set_yticklabels((0,Int64(gate_time))) #Time-scale
    end


    function Ψ_x(psi::Array{ComplexF64,1}, dims::Int, L::Float64, range_x::Array{Float64,1})
        global Ψ_x = zeros(length(range_x)); global i=0;
        for x in range_x
            i+=1;
            ϕ_x = zeros(dims);
            for n in 0:dims-1
                H_n=Fun(GaussWeight(Hermite(L), L/2), [zeros(n);1]);
                ϕ_x[n+1]=sqrt(1/(2^n*factorial(big(n))))*(L/pi)^(0.25)*H_n(x);
            end
            Ψ_x[i] = psi ⋅ ϕ_x
        end
        fig, ax = plt.subplots();
        psi_x = ax.scatter(range_x, (abs.(Ψ_x))^2);
        xlabel("\$ x \$"); ylabel("\$ |\\Psi_x|^2 \$")
    end


    function BasisProjection(basis_states::Array, eigen_states::Array; labels=nothing)
        num_states = size(eigen_states)[2]
        num_basis = size(basis_states)[2]
        rcParams = PyDict(matplotlib."rcParams");
        rcParams["figure.figsize"] = [num_basis/2,num_states/2+1]
        fig, ax1 = plt.subplots();
        mat = ax1.matshow(round.([abs(basis_states[:,j] ⋅ eigen_states[:,i]) for i in 1:num_states, j in 1:num_basis],digits=2),cmap="Blues");
        ax1.set_xticks(collect(0:1:num_basis-1));
        ax1.set_xticklabels(labels,minor=true,rotation="90");
        ax1.set_xticklabels(labels,rotation="90");
        ax1.set_yticks(collect(-0.5:1:num_states));
        ax1.set_yticklabels(["\$\\phi\$"*string(i) for i in 1:num_states],minor=true);
        ax1.set_yticklabels(["\$\\phi\$"*string(i) for i in 1:num_states]);
        fig.colorbar(mat,orientation="horizontal")
    end # function


    # See https://github.com/qutip/qutip/blob/master/qutip/bloch.py
    function BlochSphere(set_psi::Array{Array{Complex{Float64},1},1}, hamiltonian::Array{Complex{Float64},2}, sigma_x::SparseMatrixCSC{Complex{Float64},Int64} , sigma_y::SparseMatrixCSC{Complex{Float64},Int64} , sigma_z::SparseMatrixCSC{Complex{Float64},Int64}, gate_time::Float64, steps::Int, labels; start_time=0)
        uf = collect(0:pi/100:pi+0.01);vf = collect(0:pi/100:pi+0.01);
        ub = collect(-pi+0.01:pi/100:0);vb = collect(0:pi/100:pi+0.01);
        xf = cos.(uf)*sin.(vf)';yf = sin.(uf)*sin.(vf)';zf=ones(length(uf))*cos.(vf)';
        xb = cos.(ub)*sin.(vb)';yb = sin.(ub)*sin.(vb)';zb=ones(length(ub))*cos.(vb)';

        fig = plt.figure(figsize=(7,7));ax = fig.add_subplot(111, projection="3d");ax.set_axis_off();ax.view_init(elev=15., azim=90.)

        ax.set_xlim3d(-1.0, 1.0);ax.set_ylim3d(-1.0, 1.0);ax.set_zlim3d(-1.0, 1.0);

        #ax.plot_surface(xf, yf, zf, rstride=2, cstride=2, color="#FFDDDD", linewidth=0,alpha=0.2);ax.plot_surface(xb, yb, zb, rstride=2, cstride=2, color="#FFDDDD", linewidth=0,alpha=0.2);
        #ax.plot_wireframe(xf, yf, zf, rstride=10, cstride=10, color="gray", alpha=0.2, linewidth=1);ax.plot_wireframe(xb, yb, zb, rstride=10, cstride=10, color="gray", alpha=0.2, linewidth=1);

        ax.plot([-1,1],[0,0],zs=0,color="black", alpha=0.5, linewidth=2, marker="o");ax.plot([0,0],[-1,1],zs=0,color="black", alpha=0.5, linewidth=2, marker="o");ax.plot([0,0],[-1,1],zs=0,zdir="x",color="black", alpha=0.5, linewidth=2, marker="o");

        ax.plot(cos.(ub),sin.(ub),zs=0,zdir="z",color="black",alpha=0.3,linewidth=1.5);ax.plot(cos.(ub),sin.(ub),zs=0,zdir="x",color="black",alpha=0.3,linewidth=1.5);ax.plot(cos.(ub),sin.(ub),zs=0,zdir="y",color="black",alpha=0.3,linewidth=1.5);ax.plot(cos.(uf),sin.(uf),zs=0,zdir="x",color="black",alpha=0.3,linewidth=1.0);    ax.plot(cos.(uf),sin.(uf),zs=0,zdir="y",color="black",alpha=0.3,linewidth=1.0);;ax.plot(cos.(uf),sin.(uf),zs=0,zdir="z",color="black",alpha=0.3,linewidth=1.0);

        ax.text(0,1.1,0, "\$y\$", size=32);ax.text(1.6,0,0, "\$x\$", size=32);ax.text(0,0,1.1, "\$z\$", size=32)

        #ax.text(0,0,1.2, "\$|0\\rangle\$");ax.text(0,0,-1.5, "\$|1\\rangle\$");
        colormaps = ["Reds","Blues"]
        n=0;
        for psi_i in set_psi
            n+=1;
            psi_t = expv_timestep(collect(start_time:gate_time/steps:gate_time),-im*hamiltonian,psi_i);
            expv_x = real.([psi_t[:,t]'*(sigma_x)*psi_t[:,t] for t in 1:size(psi_t)[2]]);
            expv_y = real.([psi_t[:,t]'*(sigma_y)*psi_t[:,t] for t in 1:size(psi_t)[2]]);
            expv_z = real.([psi_t[:,t]'*(sigma_z)*psi_t[:,t] for t in 1:size(psi_t)[2]]);
            ax.scatter3D(expv_x,expv_y,expv_z,c=collect(0:1/(size(psi_t)[2]-1):1),cmap=colormaps[n],s=2.0,label=labels[n])
            ax.plot3D(expv_x,expv_y,expv_z,linewidth=3.0,label=labels[n])
            return expv_x, expv_y, expv_z
        end
        legend()

    end


end
