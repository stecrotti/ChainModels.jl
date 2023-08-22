marg = @inferred marginals(chain)
nmarg = @inferred neighbor_marginals(chain)
pmarg = @inferred pair_marginals(chain)

P = [evaluate(chain, x) for x in Iterators.product((1:q for q in qs)...)]
@testset "normalization" begin
    @test sum(P) ≈ (@inferred normalization(chain))
    P ./= sum(P)
    @test normalization(normalize(chain)) ≈ 1
end

@testset "Marginals" begin
    for i in 1:L
        @test vec(marg[i]) ≈ vec( sum(P, dims=(1:L)[Not(i)]) )
    end
end

@testset "Neighbor marginals" begin
    for i in 1:L-1
        @test Matrix(nmarg[i]) ≈ reshape(sum(P, dims=(1:L)[Not(i:i+1)]), qs[i], qs[i+1])
        @test nmarg[i] ≈ pmarg[i,i+1]
    end
end

@testset "Pair marginals" begin
    for i in 1:L-1
        for j in i+1:L
            @test pmarg[i,j] ≈ reshape(sum(P, dims=(1:L)[Not(i,j)]), qs[i], qs[j])
        end
    end
end

@testset "Energy" begin
    E = @inferred energy(chain)
    E_exhaust = sum(-logevaluate(chain, x)*p for (x,p) in pairs(P))
    @test E ≈ E_exhaust
end