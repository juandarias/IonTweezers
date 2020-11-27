module plotting_functions
    #__precompile__(false)

    using DrWatson
    @quickactivate "IonTweezers"
    using LinearAlgebra, PyPlot, SparseArrays, PyCall
    include(plotsdir()*"/format_Plot.jl")

    export Crystal2D, PlotMatrix, GraphCoupling, PlotSpectra, PlotMatrixArticle, PlotUnitary, TweezerStrength

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

    function PlotMatrixArticle(interaction_matrix::Array{Float64,2}; save_results=false, name_figure=nothing)
        fig, ax = plt.subplots()
        Nions = size(interaction_matrix)[1]
        mat = ax.matshow(interaction_matrix, cmap="seismic")
        cbar = fig.colorbar(mat)
        lim_color = maximum(abs.(interaction_matrix))
        mat.set_clim(vmin=-lim_color,vmax=lim_color)
        ax.set_xticklabels(collect(0:1:Nions-1))
        ax.set_yticklabels(collect(0:1:Nions-1))
        cbar.set_label("\$ J_{i,j}/(2\\pi)\\, \$(Hz)")
        xlabel("Ion i")
        ylabel("Ion j")
        #ax.xaxis.set_label_position("top")
        ax.xaxis.tick_bottom()
        save_results == true && savefig(plotsdir(name_figure)*".svg", bbox_inches="tight")
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

end
