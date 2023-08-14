@testset "Periodic BC" begin
    L = length(chain)
    P = [evaluate(chain, x) for x in Iterators.product((1:q for q in nstates(chain))...)]
    @testset "normalization" begin
        @test sum(P) ≈ (@inferred normalization(chain))
        P ./= sum(P)
        @test normalization(normalize(chain)) ≈ 1
    end

    marg = @inferred marginals(chain)
    nmarg = @inferred neighbor_marginals(chain)
    pmarg = @inferred pair_marginals(chain)

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
end

@testset "Periodic BC" begin
    L = length(pchain)
    P = [evaluate(pchain, x) for x in Iterators.product((1:q for q in nstates(pchain))...)]
    @testset "normalization" begin
        Z = @inferred normalization(pchain) 
        @test sum(P) ≈ Z
        P ./= sum(P)
        @test normalization(normalize(pchain)) ≈ 1
        l = accumulate_left(pchain)
        r = accumulate_right(pchain)
        qc, c = ChainModels.findcenter(pchain.f)
        for i in (1:L)
            @test Z ≈ sum(exp, l[i-1] .+ r[i+1]') 
        end
    end
    
    marg = @inferred marginals(pchain)
    nmarg = @inferred neighbor_marginals(pchain)
    # pmarg = @inferred pair_marginals(pchain)

    @testset "Marginals" begin
        for i in 1:L
            @test vec(marg[i]) ≈ vec( sum(P, dims=(1:L)[Not(i)]) )
        end
    end
    
    @testset "Neighbor marginals" begin
        for i in 1:L-1
            @test nmarg[i] ≈ reshape(sum(P, dims=(1:L)[Not(i,i+1)]), qs[i], qs[i+1])
            # @test nmarg[i] ≈ pmarg[i,i+1]
        end
        @test nmarg[end]'[:] ≈ sum(P, dims=(1:L)[Not(L,1)])[:]
    end
    
    
end