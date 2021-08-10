module plotting_functions
    #__precompile__(false)

    using DrWatson
    @quickactivate "IonTweezers"
    using LinearAlgebra, PyPlot, SparseArrays, PyCall
    include(plotsdir()*"/format_Plot.jl")

    export Crystal2D, PlotMatrix, GraphCoupling, PlotSpectra, PlotMatrixArticle, PlotUnitary, TweezerStrength, ModesInPlane, PlotAnyMatrix, PlotMatrix3D

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
        ax.set_xticklabels(collect(0:1:Nions-1))
        ax.set_yticklabels(collect(0:1:Nions-1))
        cbar.set_label("\$ J_{i,j} \$[Hz]")
        xlabel("Ion i")
        ylabel("Ion j")
        ax.xaxis.set_label_position("top")
    end

    function PlotMatrixArticle(interaction_matrix::Array{Float64,2}; save_results=false, name_figure=nothing, location=nothing, comment="")
        fig, ax = plt.subplots()
        Nions = size(interaction_matrix)[1]
        mat = ax.matshow(interaction_matrix, cmap="seismic")
        cbar = fig.colorbar(mat)
        lim_color = maximum(abs.(interaction_matrix))
        mat.set_clim(vmin=-lim_color,vmax=lim_color)
        ax.set_xticks(0:1:Nions-1);ax.set_yticks(0:1:Nions-1)
        ax.set_xticklabels(collect(1:1:Nions))
        ax.set_yticklabels(collect(1:1:Nions))
        cbar.set_label("\$ J_{i,j}/(2\\pi)\\, \$(Hz)")
        xlabel("Ion i")
        ylabel("Ion j")
        PyPlot.text(-1,-1.25,comment)
        #ax.xaxis.set_label_position("top")
        ax.xaxis.tick_bottom()
        save_results == true && savefig(plotsdir(location, name_figure)*".svg", bbox_inches="tight")
    end

    function PlotAnyMatrix(any_matrix::Array{Float64,2}; save_results=false, name_figure=nothing, location=nothing, label_x="x", label_y="y", label_c="c", comment="", label_vals=nothing)
        fig, ax = plt.subplots()
        Nions = size(any_matrix)[1]
        mat = ax.matshow(any_matrix, cmap="seismic")
        cbar = fig.colorbar(mat)
        lim_color = maximum(abs.(any_matrix))
        mat.set_clim(vmin=-lim_color,vmax=lim_color)
        ax.set_xticks(0:1:Nions-1);ax.set_yticks(0:1:Nions-1)
        ax.set_xticklabels(collect(1:1:Nions))
        ax.set_yticklabels(collect(1:1:Nions))
        xlabel(label_x)
        ylabel(label_y)
        cbar.set_label(label_c)
        text(-1,-1.25,comment)
        #ax.xaxis.set_label_position("top")
        ax.xaxis.tick_bottom()
        if label_vals !== nothing
            for i=0:Nions-1, j=0:Nions-1
                ax.text(i,j,label_vals[i+1,j+1], ha="center", va="center")
            end
        end
        save_results == true && savefig(plotsdir(location, name_figure)*".svg", bbox_inches="tight")
    end

    function PlotMatrix3D(interaction_matrix::Array{Float64,2}; save_results=false, name_figure=nothing, location=nothing, alpha_l=1.0)
        Nions = size(interaction_matrix)[1]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=30., azim=20.)
        ls=PyPlot.colorsm.LightSource(azdeg=60, altdeg=30, hsv_min_val=0, hsv_max_val=1, hsv_min_sat=1, hsv_max_sat=0)
        norm = PyPlot.colorsm.Normalize(vmin=-1.0, vmax=1.0, clip=true)
        mapper = PyPlot.cm.ScalarMappable(norm=norm, cmap=PyPlot.cm.RdBu_r)
        crange=replace!(vcat(interaction_matrix...),0.0=>interaction_matrix[Nions,1])
        colors_bars = vcat([collect(mapper.to_rgba(value))' for value in crange]...);
        map(x-> crange[x]==0 && (colors_bars[x,4]=0),collect(1:Nions^2))
        #colors_bars = PyPlot.cm.RdBu_r(crange); #https://stackoverflow.com/questions/28752727/map-values-to-colors-in-matplotlib
        #colors_bars=PyPlot.cm.seismic(crange);
        x_grid = vcat([i for i=1:Nions, j=1:Nions]...)
        y_grid = vcat([j for i=1:Nions, j=1:Nions]...)
        
        z_height = abs.(interaction_matrix);
        z_height_up = vcat(collect(UpperTriangular(z_height))...)
        z_height_lower = vcat(collect(LowerTriangular(z_height))...)
        ax.bar3d(x_grid,y_grid,zeros(Nions^2),1,1,z_height_up, color=colors_bars, shade=true, zsort="max", lightsource=ls)
        
        ax.bar3d(x_grid,y_grid,zeros(Nions^2),1,1,z_height_lower, color=colors_bars, shade=true, zsort="max", lightsource=ls)
        ax.set_zticks([0.0,1.0])
        ax.set_yticks(collect(2:2:Nions))
        ax.set_xticks(collect(2:2:Nions))
        ax.zaxis.set_rotate_label(false); ax.yaxis.set_rotate_label(false); ax.xaxis.set_rotate_label(false) 
        ax.set_xlabel("Ion i");ax.set_ylabel("Ion j");ax.set_zlabel("\$\\log\\left(\\frac{J^{(i,j)}}{J^{(i,j)}_{\\textrm{min}}}\\right)\$")
        ax.grid(false)
        save_results == true && savefig(plotsdir(location, name_figure)*".svg", bbox_inches="tight")
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


    function GraphCouplingError(error_interaction_matrix::Array, ion_positions::Array; plane="XZ", upper_cutoff::Float64=1000.0, lower_cutoff::Float64=0.0, zero_offset::Float64=0.0)
        fig, ax = plt.subplots()
        Nions = size(ion_positions,2)
        ### Position ions and edges
        plane == "XZ" && (posx = ion_positions[1,:])
        plane == "YZ" && (posx = ion_positions[2,:])
        posz = ion_positions[3,:]
        xquiver = vcat([[posx[j] for i in j+1:Nions] for j in 1:Nions]...)
        zquiver = vcat([[posz[j] for i in j+1:Nions] for j in 1:Nions]...)
        uquiver = vcat([[-posx[j] + posx[i] for i in j+1:Nions] for j in 1:Nions]...)
        vquiver = vcat([[-posz[j] + posz[i] for i in j+1:Nions] for j in 1:Nions]...)
        dist = [norm([uquiver[i],vquiver[i]]) for i in 1:length(xquiver)]

        ### Colors of coupling edges
        color_edge = vcat([[error_interaction_matrix[i,j] for i in j+1:Nions] for j in 1:Nions]...)

        ### Remove non-coupling egdes
        index_zero=findall(color_edge.<=zero_offset)
        deleteat!(xquiver,index_zero);
        deleteat!(zquiver,index_zero);    
        deleteat!(uquiver,index_zero);
        deleteat!(vquiver,index_zero);
        deleteat!(color_edge,index_zero);
        deleteat!(dist,index_zero);


        #Remove non NN
        nnneighbour = [lower_cutoff > dist[i] || dist[i] > upper_cutoff for i in 1:length(dist)]
        deleteat!(xquiver,nnneighbour);
        deleteat!(zquiver,nnneighbour);    
        deleteat!(uquiver,nnneighbour);
        deleteat!(vquiver,nnneighbour);
        deleteat!(color_edge,nnneighbour)

        ### Creates figure
        Q= ax.quiver(xquiver,zquiver,uquiver,vquiver, color_edge, angles="xy", scale_units="xy", scale=1.0, headaxislength=0, headlength=0, headwidth = 1, cmap="Reds", alpha=0.75)
        
        max_error = maximum(error_interaction_matrix)
        Q.set_clim(vmin=0,vmax=max_error)
        #Q.set_clim(vmin=-1,vmax=1)
        #plt.xlim(minimum(posz), maximum(posz))
        fig.colorbar(Q,label="\$\\epsilon(J_{i,j})\$")
        ax.scatter(posx,posz,s=50, c="green")
        for i in 1:Nions
            ax.annotate(string(i),(posx[i],posz[i]),fontsize="small")
        end
        #plt.figure(frameon=false)
        ax.set_aspect(1)
        ax.axis("off")
    end


    function GraphCoupling(interaction_matrix::Array, ion_positions::Array; plane="XZ", upper_cutoff::Float64=1000.0, lower_cutoff::Float64=0.0, zero_offset::Float64=0.0, label="\$\\tilde J_{i,j}\$",title_plot=false)
        fig, ax = plt.subplots()
        Nions = size(ion_positions,2)
        ### Position ions and edges
        plane == "XZ" && (posx = ion_positions[1,:])
        plane == "YZ" && (posx = ion_positions[2,:])
        posz = ion_positions[3,:]
        xquiver = vcat([[posx[j] for i in j+1:Nions] for j in 1:Nions]...)
        zquiver = vcat([[posz[j] for i in j+1:Nions] for j in 1:Nions]...)
        uquiver = vcat([[-posx[j] + posx[i] for i in j+1:Nions] for j in 1:Nions]...)
        vquiver = vcat([[-posz[j] + posz[i] for i in j+1:Nions] for j in 1:Nions]...)
        dist = [norm([uquiver[i],vquiver[i]]) for i in 1:length(xquiver)]

        ### Colors of coupling edges
        color_edge = vcat([[interaction_matrix[i,j] for i in j+1:Nions] for j in 1:Nions]...)

        ### Remove non-coupling egdes
        index_zero=findall(abs.(color_edge).<=zero_offset)
        deleteat!(xquiver,index_zero);
        deleteat!(zquiver,index_zero);    
        deleteat!(uquiver,index_zero);
        deleteat!(vquiver,index_zero);
        deleteat!(color_edge,index_zero);
        deleteat!(dist,index_zero);


        #Remove non NN
        nnneighbour = [lower_cutoff > dist[i] || dist[i] > upper_cutoff for i in 1:length(dist)]
        deleteat!(xquiver,nnneighbour);
        deleteat!(zquiver,nnneighbour);    
        deleteat!(uquiver,nnneighbour);
        deleteat!(vquiver,nnneighbour);
        deleteat!(color_edge,nnneighbour)

        ### Creates figure
        Q= ax.quiver(xquiver,zquiver,uquiver,vquiver, color_edge, angles="xy", scale_units="xy", scale=1.0, headaxislength=0, headlength=0, headwidth = 1, cmap="RdBu", alpha=0.75)
        
        
        Q.set_clim(vmin=-1,vmax=1)
        #plt.xlim(minimum(posz), maximum(posz))
        fig.colorbar(Q,label=label)
        ax.scatter(posx,posz,s=50, c="green")
        for i in 1:Nions
            ax.annotate(string(i),(posx[i],posz[i]),fontsize="small")
        end
        #plt.figure(frameon=false)
        ax.set_aspect(1)
        ax.axis("off")
        title_plot != false && ax.set_title(title_plot)
    end

    function Crystal2D(ion_positions::Array, tweezer_freq::Array; plane="XZ", offset_label=0.01)
        fig, ax = plt.subplots()
        Nions = size(ion_positions,2);
        plane == "XZ" && (posx = ion_positions[1,:])
        plane == "YZ" && (posx = ion_positions[2,:])
        posz = ion_positions[3,:]
        ts = ax.scatter(posx,posz,c=abs.(tweezer_freq),s=200,cmap="Reds")
        for i in 1:Nions
            ax.annotate("\$\\;\$"*string(i),(posx[i] + offset_label, posz[i] + offset_label),fontsize="small")
        end
        ax.axis("off")
        fig.colorbar(ts, label="\$ \\Omega_p/\\omega_z\$", orientation="horizontal",fraction=0.046, pad=0.1)
        ax.set_aspect(1)
    end

    function ModesInPlane(pos_ions::Array{Float64,2}, modes_x::Array{Float64,2}, modes_z::Array{Float64,2}, freqs::Array{Float64,1}; scale::Float64=1.0, mode_num::Array{Int64,1}=collect(1:Nions), plane="XZ")
        Nions = size(pos_ions, 2)
        plots_modes=[];
        plane == "XZ" && (posx = pos_ions[1,:])
        plane == "YZ" && (posx = pos_ions[2,:])
        posz = pos_ions[3,:]
        bmx(n)=scale*modes_x[:,n]
        bmz(n)=scale*modes_z[:,n]
        for n in mode_num
            #default(titlefont=(10, "times"))
            plot_n = Plots.scatter(posz,posx,markersize=5,label="", title="ω (MHz)="*string(round(freqs[n],digits=2)),ticks=false)
            vecz_plot = Plots.quiver!(posz,posx,quiver=(bmz(n), zeros(Nions)),aspect_ratio=1.0,width=2.0,ticks=false)
            vecx_plot = Plots.quiver!(posz,posx,quiver=(zeros(Nions), bmx(n)),aspect_ratio=1.0,width=2.0,ticks=false)
            Plots.xaxis!(showaxis=false);Plots.xgrid!(false);Plots.ygrid!(false);
            push!(plots_modes, plot_n)
            #push!(plots_modes, vec_plot)
        end
        return Plots.plot(plots_modes...)
    end

    
    function ModesIn3D(pos_ions::Array{Float64,2}, modes_x::Array{Float64,2}, modes_y::Array{Float64,2}, modes_z::Array{Float64,2}, freqs::Array{Float64,1}; scale::Float64=1.0, mode_num::Array{Int64,1}=collect(1:Nions))
        Nions = size(pos_ions, 2)
        plots_modes=[];
        posx = pos_ions[1,:]
        posy = pos_ions[2,:]
        posz = pos_ions[3,:]
        bmx(n)=scale*modes_x[:,n]
        bmy(n)=scale*modes_y[:,n]
        bmz(n)=scale*modes_z[:,n]
        for n in mode_num
            #default(titlefont=(10, "times"))
            plot_n = Plots.scatter(posx,posy,posz,markersize=5,label="", title="ω (MHz)="*string(round(freqs[n],digits=2)))
            vec_plot = Plots.quiver!(posx,posy,posz,quiver=(bmx(n),bmy(n),bmz(n)),width=2.0)
            #Plots.xaxis!(showaxis=false);Plots.xgrid!(false);Plots.ygrid!(false);
            push!(plots_modes, plot_n)
            #push!(plots_modes, vec_plot)
        end
        return Plots.plot(plots_modes...)
    end

    function ModeStrength(ion_positions::Array, mode_amplitude::Array; plane="XZ", offset_label=0.01)
        fig, ax = plt.subplots()
        Nions = size(ion_positions,2);
        plane == "XZ" && (posx = ion_positions[1,:])
        plane == "YZ" && (posx = ion_positions[2,:])
        posz = ion_positions[3,:]
        ms = ax.scatter(posx,posz,c = mode_amplitude,s=200,cmap="PiYG")
        ms.set_clim(vmin=-1,vmax=1)
        for i in 1:Nions
            ax.annotate("\$\\;\$"*string(i),(posx[i] + offset_label, posz[i] + offset_label),fontsize="x-small")
        end
        ax.axis("off")
        fig.colorbar(ms, label="\$ b_m^{(i)}\$", orientation="horizontal",fraction=0.046, pad=0.1)
        ax.set_aspect(1)
    end

    function TweezerStrength(ion_positions::Array, tweezer_freq::Array; save_results=false, name_figure=nothing, location=nothing)
        Nions = size(ion_positions)[2];
        posx = ion_positions[1,:]
        posz = ion_positions[3,:]
        twe=plt.scatter(posz,posx,c=abs.(tweezer_freq),s=200,cmap="Reds")
        plt.figure(frameon=false)
        plt.axis("off")
        twe.set_clim(vmin=0,vmax=maximum(tweezer_freq))
        plt.colorbar(twe,label="\$ \\Omega_p/\\omega_z\$", orientation="horizontal")
        twe=plt.scatter(posz,posx,c=abs.(tweezer_freq),s=200,cmap="Reds")
        #annotate(string(1),(posz[1],0.01+posx[1]))
        #annotate(string(Nions),(posz[Nions],0.01+posx[Nions]))
        
        save_results == true && savefig(plotsdir(location, name_figure)*".svg", bbox_inches="tight")
    end

    function PlotSpectra(phonon_frequencies::Array, beat_note::Float64; unpinned_frequencies=nothing)
        figure()
        modes = length(phonon_frequencies)
        #modes_plane = Int(modes*2/3);
        unpinned_frequencies!=nothing && ((markerline_u, stemlines_u, baseline_u) = stem(unpinned_frequencies,collect(1:1:modes), linefmt="C7:", markerfmt="C7h",basefmt="None"))
        markerline, stemlines, baseline = stem(phonon_frequencies[1:modes_plane],collect(1:1:modes_plane), linefmt="b-", markerfmt="bh",basefmt="None")
        markerline_t, stemlines_t, baseline_t = stem(phonon_frequencies[modes_plane+1:modes],collect(modes_plane+1:1:modes), linefmt="g-", markerfmt="gh",basefmt="None")
        stem([beat_note],[22],markerfmt="C3o",linefmt="red")
        plt.setp(stemlines, linewidth=1)
        plt.setp(stemlines_t, linewidth=1)
        unpinned_frequencies!=nothing && plt.setp(stemlines_u, linewidth=0.5)
        xlabel("\$\\omega_m/2\\pi\$ [MHz]")
        ylabel("Mode")
        #xscale("log")
    end

    function PlotSpectraSingle(phonon_frequencies::Array, beat_note::Float64; unpinned_frequencies=nothing)
        figure()
        modes = length(phonon_frequencies)
        unpinned_frequencies!=nothing && ((markerline_u, stemlines_u, baseline_u) = stem(unpinned_frequencies,collect(1:1:modes), linefmt="C7:", markerfmt="C7h",basefmt="None"))
        markerline_t, stemlines_t, baseline_t = stem(phonon_frequencies, collect(1:modes), linefmt="g-", markerfmt="gh",basefmt="None")
        stem([beat_note],[22],markerfmt="C3o",linefmt="red")
        plt.setp(stemlines_t, linewidth=1)
        unpinned_frequencies!=nothing && plt.setp(stemlines_u, linewidth=0.5)
        xlabel("\$\\omega_m/2\\pi\$ [MHz]")
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
