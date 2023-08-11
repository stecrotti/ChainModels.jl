@testset "PDF" begin
    x = [rand(1:q) for q in qs]
    @test pdf(chain, x) ≈ evaluate(chain, x) / normalization(chain)
end

P = [(@inferred evaluate(chain, x)) for x in Iterators.product((1:q for q in qs)...)]
P ./= sum(P)

@testset "Sampling" begin
    # test that empirical probabilities correlate better and better with true ones as the 
    #  number of samples increases
    rng = MersenneTwister(0)
    nsamples = [10^3, 10^4, 10^5]
    cors = zeros(length(nsamples))
    for i in eachindex(nsamples)
        x = [rand(rng, chain) for _ in 1:nsamples[i]]
        prop = proportionmap(x)
        a = sort(collect(values(prop)), rev=true)
        b = sort(P[:], rev=true)[1:length(a)]
        cors[i] = cor(a, b)
    end
    @test issorted(cors)
end

@testset "Mean, variance" begin
    μ = @inferred mean(chain)
    μ_exhaust = [sum(x[i]*p for (x,p) in pairs(P)) for i in 1:L]
    v = @inferred var(chain)
    v_exhaust = [sum((x[i]-μ[i])^2*p for (x,p) in pairs(P)) for i in 1:L]
    @test μ ≈ μ_exhaust
end

@testset "Covariance" begin
    c = @inferred cov(chain)
    μ = [sum(x[i]*p for (x,p) in pairs(P)) for i in 1:L]
    c_exhaust = [sum((x[i]-μ[i])*(x[j]-μ[j])*p for (x,p) in pairs(P)) for i in 1:L, j in 1:L]
    @test c ≈ c_exhaust
end

@testset "Entropy" begin
    S = @inferred entropy(chain)
    S_exhaust = sum(-log(p)*p for (x,p) in pairs(P))
    @test S ≈ S_exhaust
end

@testset "KL divergence" begin
    qs = (3, 2, 4, 5, 8)
    p = ChainModel([rand(qs[i-1],qs[i]) for i in Iterators.drop(eachindex(qs),1)])
    q = ChainModel([rand(qs[i-1],qs[i]) for i in Iterators.drop(eachindex(qs),1)])
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
    x = [rand(chain) for _ in 1:100]
    y = reduce(hcat, x)
    @test (@inferred loglikelihood(chain, x)) == (@inferred loglikelihood(chain, y))
end

@testset "Gradient of log-likelihood" begin
    x = [rand(chain) for _ in 1:10]
    ll(f) = @inferred loglikelihood(ChainModel(f), x)
    df_true = grad(forward_fdm(4, 1), ll, f)[1]
    df = @inferred loglikelihood_gradient(chain, x)
    @test df ≈ df_true
    y = reduce(hcat, x)
    @test df ≈ @inferred loglikelihood_gradient(chain, y)
end