qs = (2,3,4,5,6)
L = length(qs)
K = 4
f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
chain = KChainModel(f)
Tuple(nstates(f)) == qs

chain = rand_kchain_model(K, length(qs), qs[1])                          

K = 2
f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
l = accumulate_left(f)
kl = k_accumulate_left(f)
@test permutedims.(l) ≈ kl

r = accumulate_right(f)
kr = k_accumulate_right(f)
@test r ≈ kr

logZex = log(sum(evaluate(chain,x) for x in Iterators.product((1:q for q in qs)...)))

K = 3
f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
chain = KChainModel(f)
l = k_accumulate_left(f)
r = k_accumulate_right(f)

for i in 1:L-K+2
    @show logsumexp(l[i-1] + r[i+K-1])
end

@test reduce(logsumexp, last(l)) ≈ reduce(logsumexp, first(r))