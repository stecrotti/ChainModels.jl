f = [randn(qs[i:i+1]...) for i in 1:length(qs)-1]
chain = ChainModel(f)

@testset "broadcastable" begin
    X = collect.(eachcol(rand(chain, 10)))
    @test logpdf.(chain, X) == logpdf.((chain,), X)
end

@testset "Constructor with fields" begin
    h = [rand(qi) for qi in qs]
    p = ChainModel(f, h)

    function ev(f, h, x)
        w = 0.0
        for i in eachindex(f)
            w += f[i][x[i],x[i+1]]
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
    p = rand_chain_model(10, 4)
    @test length(p) == 10
    @test all(isequal(4), nstates(p))
end

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

@testset "Average Energy" begin
    E = @inferred avg_energy(chain)
    E_exhaust = sum(-logevaluate(chain, Tuple(x))*p for (x,p) in pairs(P))
    @test E ≈ E_exhaust
end