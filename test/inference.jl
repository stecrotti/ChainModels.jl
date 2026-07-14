for K in Ks
    @testset "K=$K" begin
        f_teacher = [randn(qs[i:i+K-1]...) for i in 1:length(qs)-K+1]
        chain_teacher = KChainModel(f_teacher)
        X = rand(chain_teacher, 10^1)
        fK, _ = ChainModels.compute_empirical_Kmarginals(X, K; qs=qs)

        chain = fit_mle(KChainModel, K, X, qs=qs)
        neig_marginals = neighbor_marginals(chain)

        @test all(1:L-K+1) do i
            isapprox(neig_marginals[i], fK[i], rtol=1e-8)
        end
    end
end