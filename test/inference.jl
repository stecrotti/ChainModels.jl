for K in Ks
    @testset "K=$K" begin
        f_teacher = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
        chain_teacher = KChainModel(f_teacher)
        X = rand(chain_teacher, 10^3)
        fK, _ = ChainModels.compute_empirical_Kmarginals(X, K; qs=qs)

        chain = fit_k_chain(X, K)
        neig_marginals = neighbor_marginals(chain)

        @test all(1:L-K+1) do i
            neig_marginals[i] ≈ fK[i]
        end

    end
end