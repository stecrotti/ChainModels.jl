qs = (4,3,1,2)
f = [randn(qs[i-1],qs[i]) for i in Iterators.drop(eachindex(qs),1)]
chain = ChainModel(f)
L = length(chain)

marg = marginals(chain)
nmarg = neighbor_marginals(chain)
pmarg = pair_marginals(chain)

P = [evaluate(chain, x) for x in Iterators.product((1:q for q in qs)...)]
@testset "normalization" begin
    @test sum(P) ≈ normalization(chain)
end
P ./= sum(P)

@testset "Normalization" begin
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