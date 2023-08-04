qs = (4,3,1,2)
f = [rand(qs[i-1],qs[i]) for i in Iterators.drop(eachindex(qs),1)]
chain = ChainModel(f)
L = length(chain)

@testset "PDF" begin
    x = [rand(1:q) for q in qs]
    @test pdf(chain, x) ≈ evaluate(chain, x) / normalization(chain)
end

P = [evaluate(chain, x) for x in Iterators.product((1:q for q in qs)...)]
@testset "normalization" begin
    @test sum(P) ≈ normalization(chain)
end
P ./= sum(P)

@testset "Mean, variance" begin
    μ = mean(chain)
    μ_exhaust = [sum(x[i]*p for (x,p) in pairs(P)) for i in 1:L]
    v = var(chain)
    v_exhaust = [sum((x[i]-μ[i])^2*p for (x,p) in pairs(P)) for i in 1:L]
    @test μ ≈ μ_exhaust
end

@testset "Covariance" begin
    c = cov(chain)
    μ = [sum(x[i]*p for (x,p) in pairs(P)) for i in 1:L]
    c_exhaust = [sum((x[i]-μ[i])*(x[j]-μ[j])*p for (x,p) in pairs(P)) for i in 1:L, j in 1:L]
    @test c ≈ c_exhaust
end

@testset "Entropy" begin
    S = entropy(chain)
    S_exhaust = sum(-log(p)*p for (x,p) in pairs(P))
    @test S ≈ S_exhaust
end
