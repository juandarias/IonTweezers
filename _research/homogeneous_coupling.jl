using DrWatson, PyPlot
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
include(srcdir()*"/constants.jl");
using optimizers, coupling_matrix, crystal_geometry, plotting_functions



"========================="
# Parameters and constants
"========================="

MHz = 1E6; μm = 1E-6;

Nions = 12; ω_trap = 2π*MHz*[2, 0.6, 0.07];
prefactor = ee^2/(4*π*ϵ0);

Chop_Sort(x) = sort(round.(x,digits=5), dims=2)

"========================="
# Positions and Hessian
"========================="

pos_ions = PositionIons(Nions,ω_trap,plot_position=false);
pos_ions=Chop_Sort(pos_ions)

hess=Hessian(pos_ions, ω_trap;planes=[2])


"=================="
# Coupling matrix
"=================="


paramsHom1D = Dict(:ωtrap => ω_trap, :Ωpin => sqrt.([8.81055,18.7972,35.5,40.716,48.6586,50.0153,50.0153,48.6586,40.716,35.5,18.7972,8.81055]), :μnote => 7.01356, :axis => "y", :homogeneous => false)

wsave(datadir("Article", savename(paramsHom1D, "bson")),paramsHom1D)




Jhom1D, λm, bm = Jexp1D(pos_ions, Ωpin1D, ω_trap, μ_note, planes=[2])

nJhom1D= Jhom1D./(maximum(abs.(Jhom1D)))


PlotMatrixArticle(nJhom1D)








mmhess =[70.27 2.48222 0.401366 0.142013 0.0681935 0.0384324 0.02385   0.0157471 0.0108247 0.00762059 0.00540334 0.00374876; 2.48222   65.7071 4.25521 0.611499 0.200306 0.0908229 0.0488529   0.0291017 0.0184852 0.012207 0.00819421 0.00540334; 0.401366   4.25521 61.9111 5.66243 0.767948 0.240704 0.105191 0.0547227   0.0315435 0.0193322 0.012207 0.00762059; 0.142013 0.611499   5.66243 59.0056 6.68083 0.872456 0.264755 0.112266 0.056642   0.0315435 0.0184852 0.0108247; 0.0681935 0.200306 0.767948   6.68083 57.0461 7.29657 0.924825 0.272755 0.112266 0.0547227   0.0291017 0.0157471; 0.0384324 0.0908229 0.240704 0.872456   7.29657 56.0604 7.50254 0.924825 0.264755 0.105191 0.0488529   0.02385; 0.02385 0.0488529 0.105191 0.264755 0.924825   7.50254 56.0604 7.29657 0.872456 0.240704 0.0908229   0.0384324; 0.0157471 0.0291017 0.0547227 0.112266 0.272755   0.924825 7.29657 57.0461 6.68083 0.767948 0.200306   0.0681935; 0.0108247 0.0184852 0.0315435 0.056642 0.112266   0.264755 0.872456 6.68083 59.0056 5.66243 0.611499   0.142013; 0.00762059 0.012207 0.0193322 0.0315435 0.0547227   0.105191 0.240704 0.767948 5.66243 61.9111 4.25521   0.401366; 0.00540334 0.00819421 0.012207 0.0184852 0.0291017   0.0488529 0.0908229 0.200306 0.611499 4.25521 65.7071   2.48222; 0.00374876 0.00540334 0.00762059 0.0108247 0.0157471   0.02385 0.0384324 0.0681935 0.142013 0.401366 2.48222 70.27]


evals1, evec1= eigen(mmhess+diagm([8.81055,18.7972,35.5,40.716,48.6586,50.0153,50.0153,48.6586,40.716,35.5,18.7972,8.81055].^2))

evals2, evec2 = eigen(hess2)

evec2-evec

round.(hess2-(hess+diagm([8.81055,18.7972,35.5,40.716,48.6586,50.0153,50.0153,48.6586,40.716,35.5,18.7972,8.81055].^2)), digits=4)


round.(evec1-bm, digits=4)

signΩ(omega) = diagm([sign(omega[i]) for i in 1:Nions])
        #Ω2pin_matrix(omega) = kron(diagm(kvec), diagm(signΩ(omega)*((omega./ω_trap[3]).^2)))
Ω2pin_matrix(omega) = diagm(signΩ(omega)*((omega./ω_trap[3]).^2))


round.(hess2-Ω2pin_matrix(Ωpin1D)-hess, digits=4)

evec2-evec1


signΩ(omega) = diagm([sign(omega[i]) for i in 1:Nions])
        #Ω2pin_matrix(omega) = kron(diagm(kvec), diagm(signΩ(omega)*((omega./ω_trap[3]).^2)))
        Ω2pin_matrix(omega) = diagm(signΩ(omega)*((omega).^2))
        Hess_pinned(omega) = hess + Ω2pin_matrix(omega)

        ### Phonon modes and frequencies
        hesspin=Hess_pinned(Ωpin1D);
        λm4, bm4 = eigen(round.(hesspinc,digits=13)); 
        λm3, bm3 = eigen(round.(hesspin, digits=13)); 

        hesspinc=hess+diagm([8.81055,18.7972,35.5,40.716,48.6586,50.0153,50.0153,48.6586,40.716,35.5,18.7972,8.81055].^2)

        Hess_pinned(Ωpin1D)
        round.(hesspin-(hess+diagm([8.81055,18.7972,35.5,40.716,48.6586,50.0153,50.0153,48.6586,40.716,35.5,18.7972,8.81055].^2)), digits=4)

        bm3[:,6:12]-bm4[:,6:12]

        λm3-evals

        bm3- bm4

        bm3[:,10]

        round.(bm4-evec,digits=6)

        using SparseArrays
        sparse(round.(hesspinc-hesspin, digits=12))

        hesspin[4,4]

        hess[4,4]

        using Arpack

        Ω2pin_matrix(Ωpin1D)-diagm([8.81055,18.7972,35.5,40.716,48.6586,50.0153,50.0153,48.6586,40.716,35.5,18.7972,8.81055].^2)