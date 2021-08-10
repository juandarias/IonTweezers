# Install MPI (https://www.mpich.org/documentation/guides/): sudo apt-get install libcr-dev mpich mpich-doc
#2 Install Valgrind: https://wiki.ubuntu.com/Valgrind
#3 Get Petsc https://www.mcs.anl.gov/petsc/download/index.html
#4 Install Petsc: ./configure --download-mpich --download-fblaslapack
#5 Install Slepc: https://slepc.upv.es/documentation/slepc.pdf. Use  export PETSC_ARCH=arch-linux-c-debug instead of the description of manual

using LinearAlgebra

using PetscWrap
using SlepcWrap

@testset "KSP{$ST}" begin
    A = PETSc.Mat(ST, 3,3)
    A_julia = zeros(ST,3,3)
    for i=1:3
      for j=1:3
        pos = (i-1)*3 + j
        val = RC(Complex(pos, pos))
        A[i, j] = val
        A_julia[i,j] = val
       end
    end
  
    A[3,2] = RC(Complex(6,6))  # make A non singular
    A_julia[3,2] = RC(Complex(6,6))
  
    b = PETSc.Vec(ST, 3, vtype=PETSc.C.VECMPI)
    b_julia = zeros(ST, 3)
    b[3] = RC(Complex(1,1))
    b_julia[3] = RC(Complex(1,1))
  
    @testset "ksp GMRES solves" begin
        kspg = PETSc.KSP(A)
        x = kspg\b
        x_julia = A_julia\b_julia
        @test x ≈ x_julia
  
        x = A\b
        @test x ≈ x_julia
  
        x = KSPSolveTranspose(kspg, b)
        x_julia = adjoint.(A_julia)\b_julia
        @test x ≈ x_julia
  
        x = KSPSolveTranspose(A, b)
        @test x ≈ x_julia
  
  
        pc   = PETSc.PC(ST,comm=comm(kspg),pc_type="jacobi")
        PETSc.chk(PETSc.C.PCSetOperators(pc.p,A.p,A.p))
        kspg = PETSc.KSP(pc)
        x = kspg\b
        x_julia = A_julia\b_julia
  
        @test x ≈ x_julia
        # fixme: test petscview without side effects, e.g.
        # via PetscViewerStringOpen
    end
  
    @testset "ksp BCGS solves" begin
        kspb = PETSc.KSP(A, ksp_type="bcgs")
        x = kspb\b
        x_julia = A_julia\b_julia
        @test x ≈ x_julia
    end
  end

using Test
using PETSc, MPI, LinearAlgebra, SparseArrays

m,n = 20,20
x = randn(n)
V = PETSc.VecSeq(x)


@test norm(x) ≈ norm(V) rtol=10eps()

S = sprand(m,n,0.1) + I
M = PETSc.MatSeqAIJ(S)

@test norm(S) ≈ norm(M) rtol=10eps()

w = M*x
@test w ≈ S*x

ksp = PETSc.KSP(M; ksp_rtol=1e-8, pc_type="jacobi", ksp_monitor=true)
#PETSc.settolerances!(ksp; rtol=1e-8)

@test PETSc.gettype(ksp) == "gmres" # default

pc = PETSc.PC(ksp)
@test PETSc.nrefs(pc) == 2
@test PETSc.gettype(pc) == "jacobi"

# create an extra handle, check ref count is incremented
pc_extra = PETSc.PC(ksp)
@test PETSc.nrefs(pc) == 3
# destroy extra handle, check ptr is set to null, ref count is decremented
PETSc.destroy(pc_extra)
@test pc_extra.ptr == C_NULL
@test PETSc.nrefs(pc) == 2

# safe to call destroy on null pointer
PETSc.destroy(pc_extra)

# set new pc, check ref counts are modified by ksp
pc_new = PETSc.PC{Float64}(MPI.COMM_SELF)
@test PETSc.nrefs(pc_new) == 1
PETSc.setpc!(ksp, pc_new)
@test PETSc.nrefs(pc_new) == 2
@test PETSc.nrefs(pc) == 1

PETSc.setpc!(ksp, pc)
@test PETSc.nrefs(pc_new) == 1
@test PETSc.nrefs(pc) == 2


y = ksp \ w
@test S*y ≈ w  rtol=1e-8


w = M'*x
@test w ≈ S'*x

y = ksp' \ w
@test S'*y ≈ w rtol=1e-8


f!(y,x) = y .= 2 .*x

M = PETSc.MatShell{Float64}(f!,10,10)

x = rand(10)

@test M*x ≈ 2x
@test PETSc.KSP(M) \ x ≈ x/2


function F!(fx, x)
  fx[1] = x[1]^2 + x[1]*x[2] - 3
  fx[2] = x[1]*x[2] + x[2]^2 - 6
end

J = zeros(2,2)
PJ = PETSc.MatSeqDense(J)
function updateJ!(x, args...)
    J[1,1] = 2x[1] + x[2]
    J[1,2] = x[1]
    J[2,1] = x[2]
    J[2,2] = x[1] + 2x[2]
end

S = PETSc.SNES{Float64}(MPI.COMM_SELF; ksp_rtol=1e-4, pc_type="none")
PETSc.setfunction!(S, F!, PETSc.VecSeq(zeros(2)))
PETSc.setjacobian!(S, updateJ!, PJ, PJ)
@test PETSc.solve!([2.0,3.0], S) ≈ [1.0,2.0] rtol=1e-4