
struct Network{T<:Tuple}
    layers::T
end

Network(layers...) = Network(layers)

# @code_lowered
(c::Network)(x) = _apply_layer_sequence(c.layers, x)
# @generated function _apply_layer_sequence(layers::Tuple{Vararg{Any,N}}, x) where {N}
#     symbols = vcat(:x, [gensym() for _ in 1:N])
#     calls = [:($(symbols[i+1]) = layers[$i]($(symbols[i]))) for i in 1:N]
#     Expr(:block, calls...)
# end
@generated function _apply_layer_sequence(layers::Tuple{Vararg{Any,N}}, input) where {N}
    intermediate_outputs = [gensym() for _ in 1:N]
    
    current_output = :input
    
    assignments = []
    for layer_index in 1:N
        next_output = intermediate_outputs[layer_index]
        push!(assignments, :($next_output = layers[$layer_index]($current_output)))
        current_output = next_output
    end
    
    Expr(:block, assignments..., current_output)
end

# https://github.com/FluxML/Flux.jl/blob/0e36af98f6fc5b7f3c95fe819a02172cfaaaf777/src/layers/basic.jl#L179
# TODO: TYPED MATRIXES INSTEAD OF VARIABLES??
struct Dense{M<:Matrix{Float32}, B<:Matrix{Float32}, F}
    weights::M
    biases::B
    activation::F

    function Dense(weights::M, bias = true, activation::F = identity) where {M<:Matrix{Float32}, F}
        b = create_bias(weights, bias, size(W,1))
        new{M, typeof(b), F}(W, b, activation)
    end
end

function create_bias(weights::AbstractArray, bias::Bool, dims::Integer...)
    bias ? fill!(similar(weights, dims...), 0) : false
end

function Dense((in, out)::Pair{<:Integer, <:Integer}, σ = identity;
    init = glorot_uniform, bias = true)
    Dense(init(out, in), bias, σ)
end

#nfan() = 1, 1 # fan_in, fan_out
#nfan(n) = 1, n # A vector is treated as a n×1 matrix
nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices

glorot_uniform(rng::AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(nfan(dims...)))
glorot_uniform(dims...) = glorot_uniform(Random.GLOBAL_RNG, dims...)


# init funcs https://fluxml.ai/Flux.jl/previews/PR1612/utilities/

# https://github.com/FluxML/Flux.jl/blob/0e36af98f6fc5b7f3c95fe819a02172cfaaaf777/src/losses/functions.jl
cross_entropy_loss(y, ŷ) = sum(Constant(-1.0) .* y .* log.(ŷ))
accuracy(m, x, y) = mean((m(x) .> 0.5) .== (y .> 0.5))


# https://github.com/FluxML/Flux.jl/blob/0e36af98f6fc5b7f3c95fe819a02172cfaaaf777/src/gradient.jl#L3
# https://github.com/FluxML/Zygote.jl/blob/1b914d994aea236bcb6d3d0cd6c099d86cede101/src/compiler/interface.jl#L152