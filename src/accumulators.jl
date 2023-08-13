# number of values each variable in the chain can take
function nstates(f::Vector{Matrix{T}}, ::Type{Open}) where {T<:Real}
    (i == length(f) + 1 ? size(f[end],2) : size(f[i],1) for i in 1:length(f)+1)
end
function nstates(f::Vector{Matrix{T}}, ::Type{Periodic}) where {T<:Real}
    (size(fᵢ, 1) for fᵢ in f)
end

function accumulate_left!(l, f::Vector{Matrix{T}}, ::Type{Open}) where {T<:Real}
    l[0] .= 0
    for i in eachindex(f)
        for xᵢ₊₁ in eachindex(l[i])
            l[i][xᵢ₊₁] = logsumexp(l[i-1][xᵢ] + f[i][xᵢ,xᵢ₊₁] for xᵢ in eachindex(l[i-1]))
        end
    end
    l
end

function accumulate_left(f::Vector{Matrix{T}}, BC::Type{Open}) where {T<:Real}
    l = OffsetArray([zeros(T, 1, q) for q in nstates(f, BC)], -1)
    accumulate_left!(l, f, BC)
end

function accumulate_right!(r, f::Vector{Matrix{T}}, BC::Type{Open}) where {T<:Real}
    r[end] .= 0
    for i in reverse(eachindex(f))
        for xᵢ in eachindex(r[i+1])
            r[i+1][xᵢ] = logsumexp(f[i][xᵢ,xᵢ₊₁] + r[i+2][xᵢ₊₁] for xᵢ₊₁ in eachindex(r[i+2]))
        end
    end
    r
end

function accumulate_right(f::Vector{Matrix{T}}, BC::Type{Open}) where {T<:Real}
    r = OffsetArray([zeros(T, q, 1) for q in nstates(f, BC)], +1)
    accumulate_right!(r, f, BC)
end

function accumulate_middle!(m, f::Vector{Matrix{T}}, BC::Type{Open}) where {T<:Real}
    for i in eachindex(f)
        m[i,i+1] .= f[i]
    end
    for j in Iterators.drop(axes(m,2), 1)
        for i in reverse(1:j-2)
            for xᵢ in axes(m[i,j], 1)
                for xⱼ in axes(m[i,j], 2)
                    m[i, j][xᵢ,xⱼ] = logsumexp(f[i][xᵢ,xᵢ₊₁] + m[i+1,j][xᵢ₊₁,xⱼ] 
                                                        for xᵢ₊₁ in axes(m[i+1,j], 1))
                end
            end
        end
    end
    m
end

function accumulate_middle(f::Vector{Matrix{T}}, BC::Type{Open}) where {T<:Real}
    m = OffsetArray([zeros(T, q1, q2) 
        for q1 in collect(nstates(f, BC))[1:end-1], q2 in collect(nstates(f, BC))[2:end]], 0, +1) 
    accumulate_middle!(m, f, BC)
end

#### PERIODIC BOUNDARY CONDITIONS
findcenter(f::Vector{Matrix{T}}) where T = findmin(nstates(f, Periodic)) 

function accumulate_left!(l, f::Vector{Matrix{T}}, ::Type{Periodic}) where {T<:Real}
    L = length(f)
    qc, c = findcenter(f)
    copyto!(l[mod1(c-1,L)], I); l[mod1(c-1,L)] .= log.(l[mod1(c-1,L)])
    for j in eachindex(f)
        i = mod1(j + c - 1, L)
        for xᵢ₊₁ in axes(l[i], 2), xc in axes(l[i], 1)
            l[i][xc,xᵢ₊₁] = logsumexp(l[mod1(i-1, L)][xc,xᵢ] + f[i][xᵢ,xᵢ₊₁] for xᵢ in axes(f[i], 1))
        end
    end
    return l
end

function accumulate_left(f::Vector{Matrix{T}}, BC::Type{Periodic}) where {T<:Real}
    qc, c = findcenter(f)
    l = circshift!([zeros(T, qc, q) for q in nstates(f, BC)], -1)
    accumulate_left!(l, f, BC)
end

function accumulate_right!(r, f::Vector{Matrix{T}}, ::Type{Periodic}) where {T<:Real}
    L = length(f)
    qc, c = findcenter(f)
    copyto!(r[mod1(c+1,L)], I); r[mod1(c+1,L)] .= log.(r[mod1(c+1,L)]) 
    for j in Iterators.take(reverse(eachindex(f)), L-1)
        i = mod1(j + c - 1, L)
        ip1 = mod1(i + 1, L)
        ip2 = mod1(i + 2, L)
        for xc in axes(r[ip1], 2), xᵢ in axes(r[ip1], 1)
            r[ip1][xᵢ,xc] = logsumexp(f[i][xᵢ,xᵢ₊₁] + r[ip2][xᵢ₊₁,xc] for xᵢ₊₁ in axes(f[i], 2))
        end
    end
    return r
end

function accumulate_right(f::Vector{Matrix{T}}, BC::Type{Periodic}) where {T<:Real}
    qc, c = findcenter(f)
    r = circshift!([zeros(T, q, qc) for q in nstates(f, BC)], +1)
    accumulate_right!(r, f, BC)
end