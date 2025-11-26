qs = (2,3,4,5,6)
L = length(qs)
K = 4
f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
chain = KChainModel(f)
Tuple(nstates(f)) == qs

chain = rand_kchain_model(K, length(qs), qs[1])                          

K = 2
f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
h = [randn(q) for q in qs]
l = accumulate_left(f)
kl = k_accumulate_left(f)
@test permutedims.(l) ≈ kl
chain = ChainModel(f, h)
kchain = KChainModel(f, h)

m_old = marginals(chain)
m_new = nbody_neighbor_marginals(K-1, kchain)
@test m_old ≈ m_new
@test m_old ≈ marginals(kchain)

nm_old = neighbor_marginals(chain)
nm_new = neighbor_marginals(kchain)
@test nm_old ≈ nm_new

@test avg_energy(kchain) ≈ avg_energy(kchain)

pair_marginals(kchain) ≈ pair_marginals(chain)


K = 3
f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
chain = KChainModel(f)
l = accumulate_left(chain)
r = accumulate_right(chain)

logZex = log(sum(evaluate(chain, x) for x in Iterators.product((1:q for q in qs)...)))

@test all(1:L-K+2) do i
    logsumexp(l[i-1] + r[i+K-1]) ≈ logZex
end

@test reduce(logsumexp, last(l)) ≈ reduce(logsumexp, first(r)) ≈ logZex


K = 1
f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
chain = KChainModel(f)
marg = marginals(chain)
@test marg ≈ [(a = exp.(fi); a ./= sum(a)) for fi in chain.f]

K = 4
n = 2
f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
chain = KChainModel(f)

@testset "Constructor with fields" begin
    K = 3
    f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
    chain = KChainModel(f)
    h = [randn(q) for q in qs]
end