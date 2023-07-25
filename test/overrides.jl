L = 5
q = 4
f = [rand(q, q) for _ in 1:(L-1)]
chain = ChainModel(f)

@testset "PDF" begin
    x = rand(1:q, L)
    @test pdf(chain, x) ≈ evaluate(chain, x) / normalization(chain)
end

P = [evaluate(chain, x) for x in Iterators.product(fill(1:q, L)...)]
@testset "normalization" begin
    sum(P) ≈ normalization(chain)
end
P ./= sum(P)

@testset "Mean variance" begin
    μ = mean(chain)
    μ_exhaust = [sum(x[i]*p for (x,p) in pairs(P)) for i in 1:L]
    v = var(chain)
    v_exhaust = [sum((x[i]-μ[i])^2*p for (x,p) in pairs(P)) for i in 1:L]
    @test μ ≈ μ_exhaust
end

@testset "Entropy" begin
    S = entropy(chain)
    S_exhaust = sum(-log(p)*p for (x,p) in pairs(P))
    @test S ≈ S_exhaust
end
