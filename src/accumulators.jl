"""
    nstates(f::AbstractVector{<:AbstractArray{<:Real}})

Returns an iterator with the number of values each variable can take.
"""
function nstates(f::AbstractVector{<:AbstractArray{T,K}}) where {T<:Real,K}
    it1 = [size(fi, 1) for fi in f]
    it2 = [size(f[end])[begin+1:end]...]
    return vcat(it1, it2)
end


# """
#     accumulate_left!(l, f::Vector{Matrix{T}}) where {T<:Real}

# In-place version of [`accumulate_left`](@ref)
# """
# function accumulate_left!(l, f::Vector{Matrix{T}}) where {T<:Real}
#     l[0] .= 0
#     for i in eachindex(f)
#         for xᵢ₊₁ in eachindex(l[i])
#             l[i][xᵢ₊₁] = logsumexp(l[i-1][xᵢ] + f[i][xᵢ,xᵢ₊₁] for xᵢ in eachindex(l[i-1]))
#         end
#     end
#     l
# end

# @doc raw"""
#     accumulate_left(f::Vector{Matrix{T}}) where {T<:Real}

# Compute the left partial normalization for the matrices in `f`
# ```math
# l_{i}(x_{i+1}) = \log\sum\limits_{x_1,\ldots,x_i}\prod\limits_{j=1}^i e^{f_j(x_j,x_{j+1})}
# ```
# """
# function accumulate_left(f::Vector{Matrix{T}}) where {T<:Real}
#     l = OffsetArray([zeros(T, 1, q) for q in nstates(f)], -1)
#     accumulate_left!(l, f)
# end

# """
#     accumulate_right!(l, f::Vector{Matrix{T}}) where {T<:Real}

# In-place version of [`accumulate_right`](@ref)
# """
# function accumulate_right!(r, f::Vector{Matrix{T}}) where {T<:Real}
#     r[end] .= 0
#     for i in reverse(eachindex(f))
#         for xᵢ in eachindex(r[i+1])
#             r[i+1][xᵢ] = logsumexp(f[i][xᵢ,xᵢ₊₁] + r[i+2][xᵢ₊₁] for xᵢ₊₁ in eachindex(r[i+2]))
#         end
#     end
#     r
# end

# @doc raw"""
#     accumulate_right(f::Vector{Matrix{T}}) where {T<:Real}

# Compute the right partial normalization for the matrices in `f`
# ```math
# r_{i}(x_{i-1}) = \log\sum\limits_{x_i,\ldots,x_L}\prod\limits_{j=i-1}^L e^{f_j(x_j,x_{j+1})}
# ```
# """
# function accumulate_right(f::Vector{Matrix{T}}) where {T<:Real}
#     r = OffsetArray([zeros(T, q, 1) for q in nstates(f)], +1)
#     accumulate_right!(r, f)
# end

"""
    accumulate_middle!(l, f::Vector{Matrix{T}}) where {T<:Real}

In-place version of [`accumulate_middle`](@ref)
"""
function accumulate_middle!(m, f::Vector{Matrix{T}}) where {T<:Real}
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

@doc raw"""
    accumulate_middle(f::Vector{Matrix{T}}) where {T<:Real}

Compute the middle partial normalization for the matrices in `f`
```math
m_{i,j}(x_i,x_j) = \log\sum\limits_{x_{i+1},\ldots,x_{j-1}}\prod\limits_{k=i}^{j-1} e^{f_k(x_k,x_{k+1})}
```
"""
function accumulate_middle(f::Vector{Matrix{T}}) where {T<:Real}
    m = OffsetArray([zeros(T, q1, q2) for q1 in collect(nstates(f))[1:end-1], q2 in collect(nstates(f))[2:end]], 0, +1) 
    accumulate_middle!(m, f)
end


"""
    accumulate_left!(l, f::AbstractVector{<:AbstractArray{<:Real,K}}) where {K}

In-place version of [`accumulate_left`](@ref)
"""
function accumulate_left!(l, f::AbstractVector{<:AbstractArray{T,K}}) where {T<:Real,K}
    l[0] .= 0
    for i in eachindex(f)
        for s in Iterators.product(axes(l[i])...)
            l[i][s...] = @views logsumexp(l[i-1][:,s[1:end-1]...] + f[i][:,s...])
        end
    end
    return l
end

# special case for K=1
function accumulate_left!(l, ::AbstractVector{<:AbstractArray{T,1}}) where {T<:Real}
    for li in l
        li .= 0
    end
    return l
end

@doc raw"""
    accumulate_left(f::Vector{Matrix{T}}) where {T<:Real}

Compute the left partial normalization for the matrices in `f`
```math
```
"""
function accumulate_left(f::AbstractVector{<:AbstractArray{T,K}}) where {T,K}
    l_ = [zeros(T, size(f[i+1])[1:end-1]...) for i in eachindex(f).-1]
    push!(l_, zeros(T, size(f[end])[2:end]...))
    l = OffsetArray(l_, -1)
    return accumulate_left!(l, f)
end

"""
    accumulate_right!(l, f::AbstractVector{<:AbstractArray{<:Real,K}}) where {K}

In-place version of [`accumulate_right`](@ref)
"""
function accumulate_right!(r, f::AbstractVector{<:AbstractArray{T,K}}) where {T<:Real,K}
    r[end] .= 0
    for i in reverse(eachindex(f))
        for s in Iterators.product(axes(r[i+K-1])...)
            r[i+K-1][s...] = @views logsumexp(f[i][s...,:] + r[i+K][s[2:end]...,:])
        end
    end
    return r
end
# special case for K=1
function accumulate_right!(r, ::AbstractVector{<:AbstractArray{T,1}}) where {T<:Real}
    for ri in r
        ri .= 0
    end
    return r
end

@doc raw"""
    accumulate_right(f::Vector{Matrix{T}}) where {T<:Real}

Compute the right partial normalization for the matrices in `f`
```math
```
"""
function accumulate_right(f::AbstractVector{<:AbstractArray{T,K}}) where {T,K}
    r_ = [zeros(T, size(f[i+1])[1:end-1]...) for i in eachindex(f).-1]
    push!(r_, zeros(T, size(f[end])[2:end]...))
    r = OffsetArray(r_, K-1)
    return accumulate_right!(r, f)
end