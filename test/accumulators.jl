L = 4
f = [rand(ULogarithmic, 2, 3), rand(ULogarithmic, 3, 4), rand(ULogarithmic, 4, 3)]
model = ChainModel(f)
l = accumulate_left(f)
r = accumulate_right(f)
m = accumulate_middle(f)

function evaluate_partial(model::ChainModel{T}, x, i, j) where T
    if i > j
        return one(T)
    end
    @assert length(x) == j - i + 2
    return evaluate(ChainModel(model.f[i:j]), x[1:j-i+2])
end

@testset "Accumulate left" begin
    l_exhaust = OffsetArray([zeros(ULogarithmic, 1, q) for q in nstates(f)], -1)
    l_exhaust[0] .= 1
    for j in Iterators.drop(eachindex(l_exhaust), 1)
        X = Iterators.product((1:q for q in Iterators.take(nstates(model), j+1))...)
        for x in X
            l_exhaust[j][last(x)] += evaluate_partial(model, x, 1, j)
        end
    end
    @test all(float.(l1) ≈ float.(l2) for (l1,l2) in zip(l_exhaust, l))
end

@testset "Accumulate right" begin
    r_exhaust = OffsetArray([zeros(ULogarithmic, q, 1) for q in nstates(f)], +1)
    r_exhaust[end] .= 1
    for i in Iterators.take(eachindex(r_exhaust), L-1)
        X = Iterators.product((1:q for q in Iterators.drop(nstates(model), i-2))...)
        for x in X
            r_exhaust[i][first(x)] += evaluate_partial(model, x, i-1, L-1)
        end
    end
    @test all(float.(r1) ≈ float.(r2) for (r1,r2) in zip(r_exhaust, r))
end

