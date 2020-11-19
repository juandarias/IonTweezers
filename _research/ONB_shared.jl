using Distributed; addprocs(5);
@everywhere using SparseArrays, SharedArrays, LinearAlgebra, QuantumInformation

@everywhere Nions =9;

@everywhere generalX(dims) = spdiagm(-1=>ones(dims-1),dims-1=>[1+0.0*im]); #shift matrix
@everywhere generalZ(dims) = spdiagm(0=>[round(exp(im*2*pi*(j/dims)),digits=10) for j in 0:dims-1]); #clock matrix


"================"
# Methods
"================"

@everywhere ⊗ = kron
@everywhere ptrace = QuantumInformation.ptrace


@everywhere function SylvesterBasis(dimensions::Int64, index::Tuple{Int64,Int64}) #Normalized basis
    XX = generalX(dimensions);
    ZZ = generalZ(dimensions);
    BE = (1/sqrt(dimensions))*(XX^index[1]*ZZ^index[2])
    return BE
end

@everywhere function SylvesterBasis(dimensions::Int64, index::Tuple{Int64,Int64}, XX::T, ZZ::T) where T<:SparseMatrixCSC{Complex{Float64},Int64} #Normalized basis
    BE = (1/sqrt(dimensions))*(XX^index[1]*ZZ^index[2])
    return BE
end

function ProcessFidelity(gate::T, unitary::T; parallel=false) where T<:SparseMatrixCSC{Complex{Float64},Int64}
    dims = size(gate)[1]; Fidelity = 0.0 + 0.0*im;
    mn_pairs = collect(Iterators.product(0:dims-1,0:dims-1));
    if parallel == false
        SumTr = 0.0 +0.0*im;
        for mn in mn_pairs
            Uj = SylvesterBasis(dims, mn);
            Ub = gate*Uj'*gate'*QuantumChannel(Uj, unitary);
            SumTr += tr(Ub);
            Uj=0.0;Ub=0.0;
        end
        Fidelity = (1/((dims+1)*dims^2))*(dims*SumTr + dims^2)
        GC.gc()
        return Fidelity
    end
    if parallel == true
        SumTr = @distributed (+) for mn in mn_pairs #memory "leak"
            Uj = SylvesterBasis(dims, mn);
            Ub=gate*Uj'*gate'*QuantumChannel(Uj, unitary);
            trUb=tr(Ub)
            Uj=0.0;Ub=0.0;
            trUb
        end
        Fidelity = (1/((dims+1)*dims^2))*(dims*SumTr + dims^2)
        GC.gc()
        return Fidelity
    end
end

function ProcessFidelityInput(gate::T, unitary::T, XX::T, ZZ::T; parallel=false) where T<:SparseMatrixCSC{Complex{Float64},Int64}
    dims = size(gate)[1]; Fidelity = 0.0 + 0.0*im;
    mn_pairs = collect(Iterators.product(0:dims-1,0:dims-1));
    if parallel == false
        SumTr = 0.0 +0.0*im;
        for mn in mn_pairs
            Uj = SylvesterBasis(dims, mn, XX, ZZ);
            Ub = gate*Uj'*gate'*QuantumChannel(Uj, unitary);
            SumTr += tr(Ub);
            Uj=0.0;Ub=0.0;
        end
        Fidelity = (1/((dims+1)*dims^2))*(dims*SumTr + dims^2)
        GC.gc()
        return Fidelity
    end
    if parallel == true
        SumTr = @distributed (+) for mn in mn_pairs #memory "leak"
            Uj = SylvesterBasis(dims, mn, XX, ZZ);
            Ub=gate*Uj'*gate'*QuantumChannel(Uj, unitary);
            trUb=tr(Ub)
            Uj=0.0;Ub=0.0;
            trUb
        end
        Fidelity = (1/((dims+1)*dims^2))*(dims*SumTr + dims^2)
        GC.gc()
        return Fidelity
    end
end

@everywhere function QuantumChannel(operator_basis::T, unitary::T; n_quanta::Int=0) where T<:SparseMatrixCSC{Complex{Float64},Int64};
    dimSS = Int64(size(operator_basis)[1])
    dimFS = Int64(size(unitary)[1]/dimSS);
    #n_quanta =0; #assuming ground state. It might be more precise to use a coherent state or thermal state.
    ρ_n = spdiagm(0=>insert!(zeros(dimFS-1),n_quanta+1,1));
    ρ_i = ρ_n ⊗ operator_basis
    #ρ_i = kron(operator_basis, ρ_n)
    #ρ_f = unitary*ρ_i*unitary'
    #channel = sparse(ptrace(Array(unitary*ρ_i*unitary'), [dimFS, dimSS], [1]));
    #ρ_n =0.0;ρ_i =0.0; GC.gc();
    return sparse(ptrace(Array(unitary*ρ_i*unitary'), [dimFS, dimSS], [1])); #tensor contractions ptrace definition. Last index is index to trace out
end

@everywhere function QuantumChannelParallel(operator_basis::T, unitary::T; n_quanta::Int=0) where T<:SparseMatrixCSC{Complex{Float64},Int64};
    dimSS = Int64(size(operator_basis)[1])
    dimFS = Int64(size(unitary)[1]/dimSS);
    #n_quanta =0; #assuming ground state. It might be more precise to use a coherent state or thermal state.
    ρ_n = spdiagm(0=>insert!(zeros(dimFS-1),n_quanta+1,1));
    ρ_i = ρ_n ⊗ operator_basis
    #ρ_i = kron(operator_basis, ρ_n)
    #ρ_f = unitary*ρ_i*unitary'
    channel = sparse(ptrace(Array(unitary*ρ_i*unitary'), [dimFS, dimSS], [1]));
    ρ_n =0.0;ρ_i =0.0; GC.gc();
    return channel #tensor contractions ptrace definition. Last index is index to trace out
end


GateTofNeg = sparse(kron(diagm(0=>append!(zeros(2^(Nions-1)-1),1)),[-1 -1.0*im ; -1.0*im -1]) + spdiagm(0=>ones(2^Nions)));
GateTofPos = sparse(kron(diagm(0=>append!(zeros(2^(Nions-1)-1),1)),[-0.5 1.0*im ; -1.0*im -1]) + spdiagm(0=>ones(2^Nions)));


@everywhere (gx = generalX(2^Nions);gz = generalZ(2^Nions));
@time ProcessFidelity(GateTofNeg, GateTofNeg)
#1044.105585 seconds (103.58 M allocations: 2.351 TiB, 11.34% gc time)
@time ProcessFidelity(GateTofNeg, GateTofNeg, parallel=true)
#387.644197 seconds (8.75 k allocations: 4.329 MiB, 0.03% gc time)
@time ProcessFidelityInput(GateTofNeg, GateTofNeg, gx, gz, parallel=true)
#386.702758 seconds (8.80 k allocations: 4.486 MiB, 0.03% gc time)

@everywhere base_directory = "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Code/Julia/2DCrystal/";
@everywhere framework_directory = base_directory*"framework/";
@everywhere push!(LOAD_PATH, framework_directory);
@everywhere using constants




function ToffoliCoM(g::Float64, Nions::Int64, label_p::String)
    f_scale = sqrt(Nions/3);
    η0=0.1/f_scale; g0=2*pi*0.001; ωf=2*pi*1;
    parms = Dict("A20_g" => Dict([("Ωr", 2*pi*0.1264912*sqrt(g)*f_scale), ("δf", 2*pi*0.02), ("g0", g0*g), ("ωf",ωf), ("η0", η0)]),
    "A20_0_neg" => Dict([("Ωr", 2*pi*0.1264912*sqrt(g)*f_scale), ("δf", -2*pi*0.02), ("g0", 0.0), ("ωf",ωf), ("η0", η0)]),
    "A50_g" => Dict([("Ωr", 2*pi*0.2*sqrt(g)*f_scale), ("δf", 2*pi*0.05), ("g0", g0*g), ("ωf",ωf), ("η0", η0)]),
    "A50_0_neg" => Dict([("Ωr", 2*pi*0.2*sqrt(g)*f_scale), ("δf", -2*pi*0.05), ("g0", 0.0), ("ωf",ωf), ("η0", η0)]),
    "A200_g" => Dict([("Ωr", 2*pi*0.4*sqrt(g)*f_scale), ("δf", 2*pi*0.2), ("g0", g0*g), ("ωf",ωf), ("η0", η0)]),
    "A200_0_neg" => Dict([("Ωr", 2*pi*0.4*sqrt(g)*f_scale), ("δf", -2*pi*0.2), ("g0", 0.0), ("ωf",ωf), ("η0", η0)]))
    return parms[label_p]
end

aa = ToffoliParmsF(1.0,5,"A20_0_neg")


function ProcessFidelityTimed(gate::T, unitary::T; parallel=false) where T<:SparseMatrixCSC{Complex{Float64},Int64}
    dims = size(gate)[1]; Fidelity = 0.0 + 0.0*im;
    ONB = @timeit to "ONB" SylvesterBasis(dims);
    parallel == false && (Fidelity = @timeit to "Fidelity" (1/((dims+1)*dims^2))*(dims*sum([tr(gate*Uj'*gate'*QuantumChannel(Uj, unitary)) for Uj in ONB])+dims^2));
    if parallel == true
        SumTr = @distributed (+) for Uj in ONB #memory "leak"
            tr(gate*Uj'*gate'*QuantumChannel(Uj, unitary))
        end
        Fidelity = (1/((dims+1)*dims^2))*(dims*SumTr + dims^2)
        GC.gc()
    end
    GC.gc()
    return Fidelity
end
