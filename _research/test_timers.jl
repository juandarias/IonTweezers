using TimerOutputs
using Optim, OptimTestProblems

const to = TimerOutput()
prob = UnconstrainedProblems.examples["Rosenbrock"];

f(x    ) =  @timeit to "f"  prob.f(x)
g!(x, g) =  @timeit to "g!" prob.g!(x, g)
h!(x, h) =  @timeit to "h!" prob.h!(x, h)

begin
reset_timer!(to)
@timeit to "Trust Region" begin
    res = Optim.optimize(f, g!, h!, prob.initial_x, NewtonTrustRegion())
end
show(to; allocations = false)
end