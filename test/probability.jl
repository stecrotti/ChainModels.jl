L = 5
q = 4
f = [rand(q, q) for _ in 1:(L-1)]
model = ChainModel(f)
x = rand(1:q, L)

@testset "PDF" begin
    @test pdf(model, x) ≈ evaluate(model, x) / normalization(model)
end
