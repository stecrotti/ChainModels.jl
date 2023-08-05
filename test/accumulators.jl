qs = (3, 2, 4)
f = [randn(qs[i-1],qs[i]) for i in Iterators.drop(eachindex(qs),1)]
chain = ChainModel(f)
L = length(chain)

l = accumulate_left(f)
r = accumulate_right(f)
m = accumulate_middle(f)

function evaluate_partial(chain::ChainModel{T}, x, i, j) where T
    if i > j
        return one(T)
    end
    @assert length(x) == j - i + 2
    return evaluate(ChainModel(chain.f[i:j]), x[1:j-i+2])
end

@testset "Accumulate left" begin
    l_exhaust = OffsetArray([zeros(1, q) for q in nstates(f)], -1)
    for i in Iterators.drop(eachindex(l_exhaust), 1)
        X = Iterators.product((1:q for q in Iterators.take(nstates(chain), i+1))...)
        for x in X
            l_exhaust[i][last(x)] += evaluate_partial(chain, x, 1, i)
        end
        l_exhaust[i] .= log.(l_exhaust[i])
    end
    @test all(l1 ≈ l2 for (l1,l2) in zip(l_exhaust, l))
end

@testset "Accumulate right" begin
    r_exhaust = OffsetArray([zeros(q, 1) for q in nstates(f)], +1)
    for i in Iterators.take(eachindex(r_exhaust), L-1)
        X = Iterators.product((1:q for q in Iterators.drop(nstates(chain), i-2))...)
        for x in X
            r_exhaust[i][first(x)] += evaluate_partial(chain, x, i-1, L-1)
        end
        r_exhaust[i] .= log.(r_exhaust[i])
    end
    @test all(float.(r1) ≈ float.(r2) for (r1,r2) in zip(r_exhaust, r))
end

@testset "Accumulate middle" begin
    m_exhaust = OffsetArray([zeros(q1, q2) for q1 in nstates(f)[1:end-1], q2 in nstates(f)[2:end]], 0, +1) 
    for i in axes(m_exhaust, 1)
        for j in i+1:L
            X = Iterators.product((1:q for q in nstates(chain)[i:j])...)
            for x in X
                m_exhaust[i,j][first(x),last(x)] += evaluate_partial(chain, x, i, j-1)
            end
            m_exhaust[i,j] .= log.(m_exhaust[i,j])
        end
    end
    @test all(float.(m1) ≈ float.(m2) for (m1,m2) in zip(m_exhaust, m))
end