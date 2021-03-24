using DrWatson
@quickactivate "IonTweezers"

push!(LOAD_PATH, srcdir())

using LinearAlgebra, SharedArrays, Optim, JLD, LineSearches #, Nabla, NLopt
using optimizers, coupling_matrix, crystal_geometry

"===================="
# Global definitions
"===================="
    #

"============================"
# Crystal geometry and modes
"============================"

    Nions = 12;
    ω_trap=[Ωx(2750,1500,2100,2*π*1E7),Ωy(2750,1500,2100,2*π*1E7),Ωz(2100)];

    ω_trap = 2*pi*1E6*[0.6, 4.0, 0.2];
    
    PositionIons(Nions,ω_trap,plot_position=true)
    pos_ions = PositionIons(Nions,ω_trap,plot_position=false);

    phmodes(pos_ions, trap_freq)  = JexpAD(pos_ions, zeros(Nions), trap_freq, 0.0)



    ωhex(Ω) = Jexp(pos_ions, Ω, ω_trap, 10.0)[2]
    bhex(Ω) = Jexp(pos_ions, Ω, ω_trap, 10.0)[3]


"============================"
# Gradients
"============================"


    using ForwardDiff, LinearAlgebra, Arpack, IterativeSolvers, Flux

    function fsvd(x::Array{Float64,1})
        M = kron(x,(x.^2)') #Build matrix with x as input
        F = svd(M) #Choose one eigenvector. Replace eigs by any other matrix decomposition method
        vec_1 = F.U[:,1] #Choose one eigenvector. Replace eigs by any other matrix decomposition method
        difx = norm(x-vec_1) #Some random function of an eigenvector
    end

    function feigs(x::Array{Float64,1})
        M = kron(x,(x.^2)') #Build matrix with x as input
        vec_1 = eigs(M)[2][:,1] #Choose one eigenvector. Replace eigs by any other matrix decomposition method
        difx = norm(x-vec_1) #Some random function of an eigenvector
    end

    function feigen(x::Array{Float64,1})
        M = kron(x,(x.^2)') #Build matrix with x as input
        vec_1 = eigen(M).vectors[:,1] #Choose one eigenvector. Replace eigs by any other matrix decomposition method
        difx = norm(x-vec_1) #Some random function of an eigenvector
    end


    dfsvd(x) = gradient(fsvd,x)[1] #using Flux.jl
    dfeigen(x) = gradient(feigen,x)[1] #using Flux.jl
    g_fsvd = x -> ForwardDiff.gradient(fsvd, x) #using ForwardDiff.jl

    dfsvd(rand(5)) #works
    dfeigen(rand(5)) #fails
    gradient(x -> fsvd(x), rand(5)) #works, using Zygote
    gradient(x -> feigs(x), rand(5)) #fails, using Zygote
    gradient(x -> feigen2(x), rand(5)) #fails, using Zygote
    g_fsvd(rand(5)) #Fails


"============="
# Tweezers 3D
"============="

    om_t = rand(3,Nions);
    om_zeros = zeros(3,Nions);

    JJ, omm, bm = Jexp3D(pos_ions, om_zeros, ω_trap/1E6, 0.90*ω_trap[3]/1E6);

    function ModesInPlane(pos_ions::Array{Float64,2}, modes::Array{Float64,2}, freqs::Array{Float64,1}; scale::Float64=1.0, mode_num::Array{Int64,1}=collect(1:Nions))
        plots_modes=[];
        for n in mode_num
            bmx(n)=scale*bm[:,n][1:1:Nions];
            bmz(n)=scale*bm[:,n][2*Nions+1:1:3*Nions];
            default(titlefont=(10, "times"))
            vec_plot = quiver(pos_ions[3,:],pos_ions[1,:],quiver=(bmz(n),bmx(n)),aspect_ratio=1.0,width=0.5,size=(1200,600))
            plot_n = scatter!(pos_ions[3,:],pos_ions[1,:],markersize=2,label="", title="\\omega (MHz)="*string(round(freqs[n],digits=2)))
            xaxis!(showaxis=false);xgrid!(false);ygrid!(false);
            push!(plots_modes, plot_n)
            #push!(plots_modes, vec_plot)
        end
        return plot(plots_modes...)
    end


    pn=ModesInPlane(pos_ions, bm, omm; scale=2.0)
