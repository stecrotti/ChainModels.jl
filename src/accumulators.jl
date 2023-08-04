function nstates(f::Vector{Matrix{T}}) where {T<:Real}
    N = length(f)
    Tuple(i == N + 1 ? size(f[end],2) : size(f[i],1) for i in 1:N+1)
end

function accumulate_left!(l, f::Vector{Matrix{T}}) where {T<:Real}
    l[0] .= 1
    for i in eachindex(f)
        mul!(l[i], l[i-1], f[i])
    end
    l
end

function accumulate_left(f::Vector{Matrix{T}}) where {T<:Real}
    l = OffsetArray([zeros(T, 1, q) for q in nstates(f)], -1)
    accumulate_left!(l, f)
end

function accumulate_right!(r, f::Vector{Matrix{T}}) where {T<:Real}
    r[end] .= 1
    for i in reverse(eachindex(f))
        mul!(r[i+1], f[i], r[i+2])
    end
    r
end

function accumulate_right(f::Vector{Matrix{T}}) where {T<:Real}
    r = OffsetArray([zeros(T, q, 1) for q in nstates(f)], +1)
    accumulate_right!(r, f)
end

function accumulate_middle!(m, f::Vector{Matrix{T}}) where {T<:Real}
    for i in eachindex(f)
        m[i,i+1] .= f[i]
    end
    for j in Iterators.drop(axes(m,2), 1)
        for i in reverse(1:j-2)
            mul!(m[i, j], f[i], m[i+1, j])
        end
    end
    m
end

function accumulate_middle(f::Vector{Matrix{T}}) where {T<:Real}
    m = OffsetArray([zeros(T, q1, q2) for q1 in nstates(f)[1:end-1], q2 in nstates(f)[2:end]], 0, +1) 
    accumulate_middle!(m, f)
end