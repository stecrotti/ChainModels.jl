for K in Ks
    @testset "K=$K" begin
        f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
        chain = KChainModel(f)



        X = [[rand(1:q) for q in qs] for _ in 1:10]

        @testset "broadcastable" begin
            @test logpdf.(chain, X) == logpdf.((chain,), X)
        end

        @testset "Type inference" begin
            @inferred evaluate(chain, X[1])
        end

        @testset "Constructor with fields" begin
            h = [rand(qi) for qi in qs]
            p = KChainModel(f, h)

            function ev(f, h, x)
                w = 0.0
                for i in eachindex(f)
                    w += f[i][x[i:i+K-1]...]
                end
                for i in eachindex(h)
                    w += h[i][x[i]]
                end
                return w
            end

            P = [logevaluate(p, x) for x in Iterators.product((1:q for q in qs)...)]
            Ptest = [ev(f, h, x) for x in Iterators.product((1:q for q in qs)...)]
            @test P ≈ Ptest
        end

        @testset "Random constructor" begin
            p = rand_k_chain_model(K, 10, 4)
            @test length(p) == 10
            @test all(isequal(4), nstates(p))
        end

        
        neigmarg = @inferred neighbor_marginals(chain)
        for n in K-1:-1:1
            @inferred nbody_neighbor_marginals(Val(n), chain)
        end
        P = [evaluate(chain, x) for x in Iterators.product((1:q for q in qs)...)]

        @testset "normalization" begin
            @test sum(P) ≈ (@inferred normalization(chain))
            @test normalization(normalize(chain)) ≈ 1
        end

        P ./= sum(P)

        @testset "Marginals" begin
            marg = @inferred marginals(chain)
            for i in 1:L
                @test marg[i] ≈ vec(sum(P, dims=(1:L)[Not(i)]))
            end
        end

        @testset "Neighbor marginals" begin
            for i in 1:L-K+1
                @test neigmarg[i] ≈ reshape(sum(P, dims=(1:L)[Not(i:i+K-1)]), qs[i:i+K-1]...)
            end
        end

        @testset "Average Energy" begin
            E = @inferred avg_energy(chain)
            E_exhaust = sum(-logevaluate(chain, Tuple(x))*p for (x,p) in pairs(P))
            @test E ≈ E_exhaust
        end
    end
end

@testset "Fully Factorized" begin
    chain = rand_factorized_model(L, 5)
    marg = marginals(chain)
    fields = [(a = exp.(fi); a ./= sum(a)) for fi in chain.f]
    @test fields ≈ marg
    @test FactorizedModel(chain.f) == chain
    h = [zeros(5) for _ in 1:L]
    @test FactorizedModel(chain.f, h) == chain
end






# qs = (2,3,4,5,6)
# L = length(qs)
# K = 4
# f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
# chain = KChainModel(f)
# Tuple(nstates(f)) == qs

# chain = rand_kchain_model(K, length(qs), qs[1])                          

# K = 2
# f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
# h = [randn(q) for q in qs]
# l = accumulate_left(f)
# kl = k_accumulate_left(f)
# @test permutedims.(l) ≈ kl
# chain = ChainModel(f, h)
# kchain = KChainModel(f, h)

# m_old = marginals(chain)
# m_new = nbody_neighbor_marginals(K-1, kchain)
# @test m_old ≈ m_new
# @test m_old ≈ marginals(kchain)

# nm_old = neighbor_marginals(chain)
# nm_new = neighbor_marginals(kchain)
# @test nm_old ≈ nm_new

# @test avg_energy(kchain) ≈ avg_energy(kchain)

# pair_marginals(kchain) ≈ pair_marginals(chain)


# K = 3
# f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
# chain = KChainModel(f)
# l = accumulate_left(chain)
# r = accumulate_right(chain)

# logZex = log(sum(evaluate(chain, x) for x in Iterators.product((1:q for q in qs)...)))

# @test all(1:L-K+2) do i
#     logsumexp(l[i-1] + r[i+K-1]) ≈ logZex
# end

# @test reduce(logsumexp, last(l)) ≈ reduce(logsumexp, first(r)) ≈ logZex


# K = 1
# f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
# chain = KChainModel(f)
# marg = marginals(chain)
# @test marg ≈ [(a = exp.(fi); a ./= sum(a)) for fi in chain.f]

# K = 4
# n = 2
# f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
# chain = KChainModel(f)

# @testset "Constructor with fields" begin
#     K = 3
#     f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
#     chain = KChainModel(f)
#     h = [randn(q) for q in qs]
# end