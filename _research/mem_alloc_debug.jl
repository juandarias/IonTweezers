using Distributed;addprocs(6);
base_directory = "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Code/Julia/2DCrystal/";
framework_directory = base_directory*"framework/";
sandbox_directory =  base_directory*"sandbox/";
output_location = base_directory*"tests/Toffoli/output/";
data_location = "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Articles/ApplPhys/Figures/";

push!(LOAD_PATH, "/mnt/c/Users/Juan/Dropbox/Projects/QuantumSimulationPhD/Results/Scripts");
push!(LOAD_PATH, framework_directory);
push!(LOAD_PATH, sandbox_directory);

using QuantumOptics.QuantumOpticsBase, SparseArrays, LinearAlgebra, ExponentialUtilities, QuantumInformation, Distributed, TimerOutputs
using constants, many_body_methods#, state_transfer
using SparseArrays, QuantumInformation, LinearAlgebra, TimerOutputs


include(framework_directory*"basis_operators.jl");
include(framework_directory*"hamiltonians.jl");
Unitary(t::Float64, Ham::SparseArrays.SparseMatrixCSC{Complex{Float64},Int64}) = sparse(exp(-im*t*Array(Ham)));
Unitary(t::Float64, Ham_Block::Array{SparseMatrixCSC{Complex{Float64},Int64},1}) = [sparse(exp(-im*t*Array(Ham_Block[i]))) for i in 1:length(Ham_Block)];


(Nions = 9; dimFS = 2; dims=2^Nions*dimFS; f_scale = sqrt(Nions/3);fbmb = FullBasis(Nions,[dimFS-1]));

(g=1;A20_g = Dict([("Ωr", 2*pi*0.1264912*sqrt(g)*f_scale), ("δf", 2*pi*0.020), ("g0", 2*pi*0.001*g), ("ωf",2*pi*1), ("η0", 0.1/f_scale)]));



GateTofNeg = sparse(kron(diagm(0=>append!(zeros(2^(Nions-1)-1),1)),[-1 -1.0*im ; -1.0*im -1]) + spdiagm(0=>ones(2^Nions)));
Udrive = Unitary(500.0/g, ToffoliHamiltonian(fbmb, A20_g, blockform=true, Ising=false, gamma_c=1.0));
UTot=RebuildUnitary(Nions,Udrive,quanta=0);


⊗ = kron
ptrace = QuantumInformation.ptrace
generalX(dims) = spdiagm(-1=>ones(dims-1),dims-1=>[1]); #shift matrix
generalZ(dims) = spdiagm(0=>[round(exp(im*2*pi*(j/dims)),digits=10) for j in 0:dims-1]);

function SylvesterBasis(dimensions::Int64) #Normalized basis
    XX = generalX(dimensions);
    ZZ = generalZ(dimensions);
    return [(1/sqrt(dimensions))*(XX^m*ZZ^n) for n in 0:dimensions-1, m in 0:dimensions-1]
end

function ProcessFidelityTimed2(gate::T, unitary::T, basisONB::B; parallel=false) where T<:SparseMatrixCSC{Complex{Float64},Int64} where B<:Array{SparseMatrixCSC{Complex{Float64},Int64},2}
    dims = size(gate)[1]; Fidelity = 0.0 + 0.0*im;
    ONB = basisONB;
    #parallel == false && (Fidelity = @timeit to "Fidelity" (1/((dims+1)*dims^2))*(dims*sum([tr(gate*Uj'*gate'*QuantumChannel(Uj, unitary)) for Uj in ONB])+dims^2));

    if parallel == false
        @timeit to "Fid" begin
        SumTr = 0.0 +0.0*im
        for Uj in ONB
            Ub = @timeit to "Ub" gate*Uj'*gate'*QuantumChannel(Uj, unitary);
            SumTr += tr(Ub);
            Ub=0.0;
        end
        Fidelity = (1/((dims+1)*dims^2))*(dims*SumTr + dims^2)
        GC.gc()
        return Fidelity
    end
    end
    if parallel == true
        SumTr = @distributed (+) for Uj in ONB #memory "leak"
            Ub=gate*Uj'*gate'*QuantumChannelParallel(Uj, unitary);
            trUb=tr(Ub)
            Ub=0.0;
            trUb
        end
        Fidelity = (1/((dims+1)*dims^2))*(dims*SumTr + dims^2)
        GC.gc()
        return Fidelity
    end

end

function QuantumChannel(operator_basis::T, unitary::T; n_quanta::Int=0) where T<:SparseMatrixCSC{Complex{Float64},Int64};
    @timeit to "QC" begin
    dimSS = Int64(size(operator_basis)[1])
    dimFS = Int64(size(unitary)[1]/dimSS);
    #n_quanta =0; #assuming ground state. It might be more precise to use a coherent state or thermal state.
    ρ_n = @timeit to "rho_n" spdiagm(0=>insert!(zeros(dimFS-1),n_quanta+1,1));
    ρ_i = @timeit to "rho_i" ρ_n ⊗ operator_basis
    #ρ_i = kron(operator_basis, ρ_n)
    #ρ_f = unitary*ρ_i*unitary'
    #channel = sparse(ptrace(Array(unitary*ρ_i*unitary'), [dimFS, dimSS], [1]));
    #ρ_n =0.0;ρ_i =0.0; GC.gc();
    return sparse(ptrace(Array(unitary*ρ_i*unitary'), [dimFS, dimSS], [1])); #tensor contractions ptrace definition. Last index is index to trace out
    end
end


const to = TimerOutput();
disable_timer!(to)
enable_timer!(to)
reset_timer!(to);

ONB = SylvesterBasis(2^Nions);

@sync @distributed for i in 1:6
    ProcessFidelityTimed2(GateTofNeg,GateTofNeg,ONB)
end

ProcessFidelityBasis(UTot,GateTofNeg,ONB, parallel=true)

show(to)



test()
GC.gc()
