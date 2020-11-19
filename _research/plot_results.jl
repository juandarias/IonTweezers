    push!(LOAD_PATH, "/mnt/c/Users/Juan/Dropbox/ProjectJuanDiego/Code/Julia/Facilitation/output/Scripts")
    
    push!(LOAD_PATH, "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Results/Scripts")
    using format_Plot
    
    
    ### Plotting result matrix
    
    soln = soln3

    Jhex(parms) = Jexp(pos_ions, parms[1:Nions], ω_trap, 8.120)[1]
    ωhex(parms) = Jexp(pos_ions, parms[1:Nions], ω_trap, 8.120)[2]
    
    Jsol = Jhex(soln)/maximum(abs.(Jhex(soln)))
    matshow(Jsol)
    colorbar(shrink=0.6)
    
    Ωmin = string(round.(soln[1:Nions]; digits=2))
    ϵ_JJ = norm(Jsol-Jstar)/norm(Jstar)
    
    text(-1,-2.5,"Ω^2[Mhz]="*Ωmin)
    text(-1,-2.0,"μ[Mhz]="*string(round(8.12, digits=4)))
    text(-1,-1.5, "ϵ="*string(round(ϵ_JJ;digits=3)))

    text(-1,7.5,"t[s]="*string(soln.time_run))
    


    figure()
    stem(ωhex(soln),collect(1:1:21),markerfmt="D",basefmt="None")
    stem([8.12],[22],markerfmt="C3o",linefmt="red")
    xlabel("f [Mhz]")
    ylabel("Mode")


    
    ### exporting matrix and minimizer
    using DelimitedFiles
    writedlm( "Jstar_AFM.csv", Jhex(soln), ',')
    writedlm( "JSG.csv", Array(JSG), ',')
    writedlm( "Jstar_FM_4.csv", Jhex(soln), ',')
    writedlm( "Jstar_AFM_minimizer.csv", soln.minimizer, ',')
    writedlm( "Jstar_FM_freqs.csv", ωhex(soln), ',')
