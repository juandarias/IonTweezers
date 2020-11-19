module diagnosis
    using Dates

    export LogMessage

    function LogMessage(message::String, location::String, testname::String)
        log = open(location*testname*".txt", "a")
        write(log, string(Dates.Time(Dates.now()))*"\n")
        write(log, message*"\n")
        close(log)
    end
end  # module diagnosis
