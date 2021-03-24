module data_export

    using DelimitedFiles, Optim, Dates

    export ExportTikzCouplingGraph, SummaryOptimization, ExportTikzMatrix, ExportTikzTweezerGraph, ExportTikzCouplingGraphError

    function ExportTikzCouplingGraph(pos_ions, target_matrix, result_matrix, name, location; threshold=0.05, plane ="YZ")
        Nions = size(pos_ions,2)
    
        ### Exporting vertices
        graph_file = Array{Any,2}(nothing,Nions+1,5)
        header = ["id" "x" "y" "color" "layer"]
        graph_file[1,:]= header
        graph_file[2:end,1] = collect(1:Nions)
        graph_file[2:end,3] = pos_ions[2,:]
        graph_file[2:end,2] = pos_ions[3,:]
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
        target_edges_color = Int.((target_edges*255).÷1)
        residual_edges_color = Int.((residual_edges*255).÷1)
    
        number_edges=count(i->(i!=0), target_edges_color)÷2
        edge_file = Array{Any,2}(nothing,number_edges+1,6)
        header = ["u" "v" "R" "G" "B" "label"]
        edge_file[1,:] = header
    
        nn=1
        for i=1:Nions, j=i+1:Nions
            if target_edges_color[i,j]!=0
                nn+=1
                edge_file[nn,1]=i
                edge_file[nn,2]=j
                edge_file[nn,6]= target_edges[i,j]
                if target_edges_color[i,j] > 0
                    edge_file[nn,3]=255;
                    edge_file[nn,4]=255 - target_edges_color[i,j];
                    edge_file[nn,5]=255 - target_edges_color[i,j]; #Assign red color
                elseif target_edges_color[i,j] < 0
                    edge_file[nn,3]=255 + target_edges_color[i,j];
                    edge_file[nn,4]=255 + target_edges_color[i,j];
                    edge_file[nn,5]=255 #Assign red color
                end
            end
        end
    
        writedlm(datadir(location, name*"edges.csv"), edge_file, ",")
    
        ### Exporting residual edges
    
        number_edges=count(i->(abs(i)>threshold*255), residual_edges_color)÷2
        edge_file_residual = Array{Any,2}(nothing,number_edges+1,5)
        header = ["u" "v" "R" "G" "B"]
        edge_file_residual[1,:] = header
    
        nn=1
        for i=1:Nions, j=i+1:Nions
            if abs(residual_edges_color[i,j])>threshold*255
                nn+=1
                edge_file_residual[nn,1]=i+Nions
                edge_file_residual[nn,2]=j+Nions
                if residual_edges_color[i,j] > 0
                    edge_file_residual[nn,3]=255;
                    edge_file_residual[nn,4]=255 - residual_edges_color[i,j];
                    edge_file_residual[nn,5]=255 - residual_edges_color[i,j]; #Assign red color
                elseif residual_edges_color[i,j] < 0
                    edge_file_residual[nn,3]=255 + residual_edges_color[i,j];
                    edge_file_residual[nn,4]=255 + residual_edges_color[i,j];
                    edge_file_residual[nn,5]=255 #Assign red color
                end
            end
        end
        writedlm(datadir(location, name*"edges_residual.csv"), edge_file_residual, ",")
    end

    function ExportTikzCouplingGraphError(pos_ions, target_matrix, target_error, residual_error, name, location; threshold=0.05, plane ="YZ")
        Nions = size(pos_ions,2)

        ### Normalize errors
        DividebyMax(matrix) = matrix./maximum(abs.(matrix))
        n_target_error = DividebyMax(target_error)
        n_residual_error = DividebyMax(residual_error)
        
    
        ### Exporting vertices
        graph_file = Array{Any,2}(nothing,Nions+1,5)
        header = ["id" "x" "y" "color" "layer"]
        graph_file[1,:]= header
        graph_file[2:end,1] = collect(1:Nions)
        graph_file[2:end,3] = pos_ions[3,:]
        graph_file[2:end,2] = pos_ions[2,:]
        graph_file[2:end,4] .= "blue"
        graph_file[2:end,5] .= 1
        writedlm(datadir(location, name*"vertices.csv"), graph_file, ",")
    
        ### Exporting second layer for residuals
        graph_file[2:end,1] = collect(1:Nions)
        graph_file[2:end,5] .= 2
        writedlm(datadir(location, name*"vertices_residual.csv"), graph_file, ",")
        
    
        ### Exporting edges
        target_edges = n_target_error
        residual_edges = n_residual_error
        target_edges_color = Int.((target_edges*255).÷1)
        residual_edges_color = Int.((residual_edges*255).÷1)
    
        number_edges=count(i->(i!=0), target_edges_color)÷2
        edge_file = Array{Any,2}(nothing,number_edges+1,5)
        #edge_file = Array{Any,2}(nothing,number_edges+1,6)
        #header = ["u" "v" "R" "G" "B" "error"]
        header = ["u" "v" "R" "G" "B"]
        edge_file[1,:] = header
    
        nn=1
        for i=1:Nions, j=i+1:Nions
            if target_edges_color[i,j]!=0
                nn+=1
                edge_file[nn,1]=i
                edge_file[nn,2]=j
                #edge_file[nn,6]= target_error[i,j]
                if target_edges_color[i,j] > 0
                    edge_file[nn,3]=255 - target_edges_color[i,j];
                    edge_file[nn,4]=255;
                    edge_file[nn,5]=255 - target_edges_color[i,j]; #Assign red color
                elseif target_edges_color[i,j] < 0
                    edge_file[nn,3]=255 + target_edges_color[i,j];
                    edge_file[nn,4]=255 + target_edges_color[i,j];
                    edge_file[nn,5]=0 #Assign red color
                end
            end
        end
    
        writedlm(datadir(location, name*"edges.csv"), edge_file, ",")
    
        ### Exporting residual edges
    
        number_edges=count(i->(abs(i)>threshold), residual_error)÷2
        edge_file_residual = Array{Any,2}(nothing,number_edges+1,5)
        #edge_file_residual = Array{Any,2}(nothing,number_edges+1,6)
        header = ["u" "v" "R" "G" "B"]
        #header = ["u" "v" "R" "G" "B" "error"]
        edge_file_residual[1,:] = header
    
        nn=1
        for i=1:Nions, j=i+1:Nions
            if abs(residual_error[i,j])>threshold
                nn+=1
                edge_file_residual[nn,1]=i
                edge_file_residual[nn,2]=j
                #edge_file_residual[nn,6]= residual_error[i,j];
                if residual_edges_color[i,j] > 0
                    edge_file_residual[nn,3]=255 - residual_edges_color[i,j];
                    edge_file_residual[nn,4]=255;
                    edge_file_residual[nn,5]=255 - residual_edges_color[i,j]; #Assign red color
                elseif residual_edges_color[i,j] < 0
                    edge_file_residual[nn,3]=255 + residual_edges_color[i,j];
                    edge_file_residual[nn,4]=255 + residual_edges_color[i,j];
                    edge_file_residual[nn,5]=0 #Assign red color
                end
            end
        end
        writedlm(datadir(location, name*"edges_residual.csv"), edge_file_residual, ",")
    end

    function SummaryOptimization(ωtrap, position_ions, hessian, target_model, objective, solution, Ωₛ, μₛ, result_matrix; note="N/A")
        
        ωtrap = ωtrap/(2π*1E6)
        method = summary(solution)
        linesearch = string(typeof(solution.method.method.linesearch!))
        minimizer = solution.minimizer
        minimum = solution.minimum
        run_time = solution.time_run

        
        summary_dict = Dict(:trap_frequency => ωtrap,
        :position_ions => position_ions,
        :hessian => Array(hessian),
        :target_model => target_model,
        :method => method,
        :objective => objective,
        :linesearch => linesearch,
        :minimizer => minimizer,
        :minimum => minimum,
        :pinning_frequency => Ωₛ,
        :beatnote => μₛ,
        :result_matrix => Array(result_matrix),
        :time => solution.time_run,
        :note => note)
    end

    function ExportTikzMatrix(result_matrix, name, location)
        Nions = size(result_matrix)[2]
        open(datadir(location, name*"_tikz_matrix.csv"), "w") do f
            first_line = prod(["(0,$i,0) (0,$i,0) " for i=0:Nions])
            write(f, chop(first_line, tail=1))
            write(f, "\n\n")
            for i=0:Nions-1
                line_upper = "($i,0,0) "
                line_lower = "($(i+1),0,0) "
                for j=0:Nions-1
                    mij = result_matrix[i+1,j+1]
                    ul = "($i,$j,$mij) "
                    ur = "($i,$(j+1),$mij) "
                    ll = "($(i+1),$(j),$mij) "
                    lr = "($(i+1),$(j+1),$mij) "
                    line_upper *= ul*ur
                    line_lower *= ll*lr
                end
                line_upper *= "($i,$Nions,0) "
                line_lower *= "($(i+1),$Nions,0) "
                write(f, chop(line_upper, tail=1))
                write(f, "\n\n")
                write(f, chop(line_lower, tail=1))
                write(f, "\n\n")
            end
            last_line = prod(["($Nions,$i,0) ($Nions,$i,0) " for i=0:Nions])
            write(f, chop(last_line, tail=1))
        end
    end

    function ExportTikzTweezerGraph(pos_ions, pinning_frequencies, name, location; plane ="YZ")
        Nions = size(pos_ions,2)
    
        ### Exporting vertices
        graph_file = Array{Any,2}(nothing,Nions+1,4)
        header = ["id" "x" "y" "color"]
        graph_file[1,:]= header
        graph_file[2:end,1] = collect(1:Nions)
        plane == "YZ" && (graph_file[2:end,3] = pos_ions[2,:])
        plane == "XZ" && (graph_file[2:end,3] = pos_ions[1,:])
        graph_file[2:end,2] = pos_ions[3,:]
        graph_file[2:end,4] = pinning_frequencies
        writedlm(datadir(location, name*"_tweezer_strength.dat"), graph_file, ",")
    end

    function ExportModeSpectra(frequency_unpinned, frequency_pinned, name, location)
        spectra_file = Array{Any,2}(nothing,length(frequency_pinned)+1,3)
        spectra_file[1,:] = ["#ωm" ,"ωm_native", "ωm_pinned"]
        spectra_file[2:end,1] = collect(1:length(frequency_pinned))
        spectra_file[2:end,2] = frequency_unpinned
        spectra_file[2:end,3] = frequency_pinned
        writedlm(datadir(location, name*"_mode_spectra.dat"), spectra_file, ",")
    end
    

end