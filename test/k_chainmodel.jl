qs = (2,3,4,5,6)
K = 4
f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
chain = KChainModel(f)
Tuple(nstates(f)) == qs

chain = rand_kchain_model(K, length(qs), qs[1])                          

K = 2
f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
l = accumulate_left(f)
kl = k_accumulate_left(f)
@assert permutedims.(l) ≈ kl

r = accumulate_right(f)
kr = k_accumulate_right(f)
@assert r ≈ kr

