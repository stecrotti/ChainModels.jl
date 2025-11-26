for K in Ks
    @testset "K=$K" begin
        f = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
        chain = KChainModel(f)

        @testset "PDF" begin
            x = [rand(1:q) for q in qs]
            @test pdf(chain, x) ≈ evaluate(chain, x) / normalization(chain)
        end

        P = [(@inferred evaluate(chain, x)) for x in Iterators.product((1:q for q in qs)...)]
        P ./= sum(P)

        @testset "Sampling" begin
            # test that empirical probabilities agree more and more with true ones as the 
            #  number of samples increases
            nsamples = [10^1, 10^3, 10^5]
            kls = zeros(length(nsamples))
            for i in eachindex(nsamples)
                x = [(@inferred rand(chain)) for _ in 1:nsamples[i]]
                prop = proportionmap(x)
                a = sort(collect(values(prop)), rev=true)
                b = sort(P[:], rev=true)[1:length(a)]
                kls[i] = kldivergence(a, b)
            end
            @test issorted(kls; rev=true)
        end

        @testset "Mean, variance" begin
            μ = @inferred mean(chain)
            μ_exhaust = [sum(x[i]*p for (x,p) in pairs(P)) for i in 1:L]
            v = @inferred var(chain)
            v_exhaust = [sum((x[i]-μ[i])^2*p for (x,p) in pairs(P)) for i in 1:L]
            @test μ ≈ μ_exhaust
        end

        if K == 2
            @testset "Covariance" begin
                c = @inferred cov(chain)
                μ = [sum(x[i]*p for (x,p) in pairs(P)) for i in 1:L]
                c_exhaust = [sum((x[i]-μ[i])*(x[j]-μ[j])*p for (x,p) in pairs(P)) for i in 1:L, j in 1:L]
                @test c ≈ c_exhaust
            end
        end

        @testset "Entropy" begin
            S = @inferred entropy(chain)
            S_exhaust = sum(-log(p)*p for p in P)
            @test S ≈ S_exhaust
        end

        @testset "KL divergence" begin
            p = KChainModel([randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1])
            q = KChainModel([randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1])
            P = [evaluate(p, x) for x in Iterators.product((1:q for q in qs)...)]
            P ./= sum(P)
            Q = [evaluate(q, x) for x in Iterators.product((1:q for q in qs)...)]
            Q ./= sum(Q)
            @test (@inferred kldivergence(p, p)) ≈ 0
            kl = (@inferred kldivergence(p, q))
            kl_exact = sum(px*log(px/qx) for (px, qx) in zip(P, Q))
            @test kl ≈ kl_exact
        end

        @testset "Log-likelihood" begin
            x = @inferred rand(chain, 100)
            y = collect(eachcol(x))
            @test (@inferred loglikelihood(chain, x)) == (@inferred loglikelihood(chain, y))
        end
    end
end