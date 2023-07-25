L = 4
q = 3
f = [rand(q, q) for _ in 1:(L-1)]
model = ChainModel(f)
marg = marginals(model)
nmarg = neighbor_marginals(model)

P = [evaluate(model, x) for x in Iterators.product(fill(1:q, L)...)]
P ./= sum(P)

@testset "Marginals" begin
    for i in 1:L
        @test vec(marg[i]) ≈ vec( sum(P, dims=(1:L)[Not(i)]) )
    end
end

@testset "Neighbor marginals" begin
    for i in 1:L-1
        @test Matrix(nmarg[i]) ≈ reshape(sum(P, dims=(1:L)[Not(i:i+1)]), q, q)
    end
end