using LinearAlgebra
using LinearAlgebra.LAPACK
using LinearAlgebra.BLAS
using Arpack


ϵ_bJ(Ω) = sum([sum((bJ[:,J] - bmprime[:,J] - sum([sum([m!=m_index[J] && (bmprime[:,J][i]*Ω[i]*bm[:,m][i])/(λmprime[J] - λm[m]) for i in 1:Nions])*bm[:,m] for m in 1:Nmodes])).^2) for J in 1:Nmodes])