
using Nabla, Random, LinearAlgebra



function f7(x) #working
    #M = kron(x,(x.^2)')
    M = (x.^2)*x'
    ev = svd(M).U #Replace eigs by any other matrix decomposition method
    vec_1 = ev[:,1]
    difx = norm(x-vec_1)
end


delf7 = ∇(f7)

for i in 1:100
    xtest = rand(5)*1000;
    #try 
        println(delf7(xtest))
    #catch
        println("fail")
    #end
end

