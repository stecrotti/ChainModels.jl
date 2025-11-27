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
    @test typeof(chain) == typeof(rand_k_chain_model(1, L, 5))
    marg = marginals(chain)
    fields = [(a = exp.(fi); a ./= sum(a)) for fi in chain.f]
    @test fields ≈ marg
    @test FactorizedModel(chain.f) == chain
    h = [zeros(5) for _ in 1:L]
    @test FactorizedModel(chain.f, h) == chain
end