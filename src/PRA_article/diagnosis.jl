module diagnosis
    using Dates, PyPlot

    export LogMessage, TracePlotter

    function LogMessage(message::String, location::String, testname::String)
        log = open(location*testname*".txt", "a")
        write(log, string(Dates.Time(Dates.now()))*"\n")
        write(log, message*"\n")
        close(log)
    end

    # Plots trace of an optimization solution from Optim. 
    function TracePlotter(solution, note)
        #fig, (ax1, ax2) = plt.subplots(2, 1)    
        trace = zeros(length(solution.trace),3)
        for n in 1:length(solution.trace)
            trace[n,1] = solution.trace[n].metadata["time"]
            trace[n,2] = solution.trace[n].value
            trace[n,3] = solution.trace[n].g_norm
        end
        trace[:,2]=trace[:,2]./maximum(trace[:,2])
        trace[:,3]=trace[:,3]./maximum(trace[:,3])
        #scatter(trace[:,1],trace[:,2], c=trace[:,1], cmap="Reds");
        #scatter(trace[:,1],trace[:,3], c=trace[:,1], cmap="Blues");
        for i in 2:length(trace[:,1])
            if trace[i,1] < trace[i-1,1]
                trace[i:end,1] .+= trace[i-1,1]-trace[i,1] 
            end
        end
        #plot(1:length(solution.trace),log.(trace[:,2]))
        #plot(1:length(solution.trace),log.(trace[:,3]))
        ax1.plot(trace[:,1],log.(trace[:,2]), label=note)
        ax2.plot(trace[:,1],log.(trace[:,3]))
        ax1.set_ylabel("ϵ")
        ax2.set_ylabel("∇ϵ")
        ax2.set_xlabel("Time (s)")
        return trace[:,1], trace[:,2], trace[:,3]
    end

    # Returns a closure over a logger, that takes a Optim trace as input. See https://philipvinc.github.io/TensorBoardLogger.jl/dev/examples/optim/
    function make_tensorboardlogger_callback(dir="log_ladder")
        logger = TBLogger(dir)

        function callback(opt_state:: Optim.OptimizationState)
            with_logger(logger) do
                @info "" opt_step = opt_state.iteration  function_value=opt_state.value gradient_norm=opt_state.g_norm time = opt_state.metadata["time"]
            end
            return false  # do not terminate optimisation
        end
        callback(trace::Optim.OptimizationTrace) = callback(last(trace))
        return callback
    end
end  # module diagnosis
