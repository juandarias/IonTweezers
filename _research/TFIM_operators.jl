J = ones(5,5);
h = ones(5);
JC = J - diagm(diag(J))


TFIM = TransverseFieldIsing(JC,h);

TFIMO = TransverseFieldIsingOld(JC,h)

TFIM_h = convert(SparseMatrixCSC, TFIM)

sparse(TFIMO.Ham - TFIM_h)

TFIM_h[1:2^4,1:2^4]

basis = generate_basis(19)

#* XY order parameter


d₀ = 1.5;
depth = 2;
pos_ions_even = TriangularLattice(d₀,depth);


GraphCoupling(zeros(19,19),pos_ions_even, plane="YZ", zero_offset=0.05);gcf()

function SublatticesABC()
    D0 = ["A"]
    D1 = ["B","C"]
    D2 = ["C", "A", "B","A"]
    for n=1:2 append!(D1, ["B","C"]); append!(D2, ["C", "A", "B","A"]); end
    return [D0... D1... D2...]
end

lattice_labels = SublatticesABC()

function ClockOrderParameter(lattice_labels, bstate)
    Oxy = 0.0
    for i in 1:length(bstate)
        lattice_labels[i] == "A" && (Oxy += bstate[i] ? 1 : -1)
        lattice_labels[i] == "B" && (Oxy += bstate[i] ? exp(im*4π/3) : -exp(im*4π/3))
        lattice_labels[i] == "C" && (Oxy += bstate[i] ? exp(-im*4π/3) : -exp(-im*4π/3))
    end
    return [Oxy, abs(Oxy)]
end

Oxy = [0,0]
for i=1:2^19
    Oxy += ClockOrderParameter(lattice_labels, basis[i])
end

input_bits = bit_rep(1,19)
Vector{Bool}(digits(1, base=2, pad=19))
nz_val = 0.0
@inbounds for site in 1:19
    @inbounds for next_site in site+1:19
        bond_J = (input_bits[site] == input_bits[next_site]) ? JAFM_19[site,next_site] : -JAFM_19[site,next_site]
        nz_val += bond_J
    end
end

TFIMO.Ham[2,2]

input_bits
psi = insert!(zeros(2^19-1),2,1)
psi'*cσᶻᵢσᶻⱼ(1,5,19)*psi
psi'*σᶻᵢ(19,19)*psi

bit_rep(1,19)
function cσᶻᵢσᶻⱼ(i,j,N)
    σᶻ = sparse([1 0; 0 -1]);
    II = sparse([1 0; 0 1]);
    i==1 || j==1 ? (op = σᶻ) : (op = II)
    for n=2:N
        n==i || n==j ? (op = kron(σᶻ, op)) : (op = kron(II,op))
    end
    return op
end

⊗ = kron
bigO = σᶻ ⊗ II

bigO*[0, 0, 1, 0]

basis19 = generate_basis(19)

basis19[1]