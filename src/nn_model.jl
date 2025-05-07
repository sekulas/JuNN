using Random: GLOBAL_RNG, AbstractRNG
include("structs.jl")
include("broadcast_operators.jl")

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
mutable struct Dense{F}
    weights::Variable
    bias::Union{Nothing, Variable}
    activation::F

    function Dense(weights::Variable, 
                   bias::Bool = false, 
                   activation::F = identity) where {F}
        b = create_bias(weights, bias)
        new{F}(weights, b, activation)
    end
end

function create_bias(weights::Variable, bias::Bool)
    bias ? Variable(fill!(similar(weights.output, size(weights.output,1)), 0), 
                    name="$(weights.name)_bias") : nothing
end

function Dense((in, out)::Pair{<:Integer, <:Integer}, activation::F=identity;
               init = glorot_uniform, 
               bias::Bool = false, 
               name="dense_$in=>$out") where {F}
    Dense(Variable(init(out, in), name=name), bias, activation)
end

function (l::Dense)(x::GraphNode)
    #TODO: Size checking    
    z = l.weights * x
    
    if !isnothing(l.bias)
        z = z .+ l.bias
    end
    
    return l.activation(z)
end

#nfan() = 1, 1 # fan_in, fan_out
#nfan(n) = 1, n # A vector is treated as a n×1 matrix
nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices

glorot_uniform(rng::AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(nfan(dims...)))
glorot_uniform(dims...) = glorot_uniform(GLOBAL_RNG, dims...)


# init funcs https://fluxml.ai/Flux.jl/previews/PR1612/utilities/

# https://github.com/FluxML/Flux.jl/blob/0e36af98f6fc5b7f3c95fe819a02172cfaaaf777/src/losses/functions.jl
accuracy(m, x, y) = mean((m(x) .> 0.5) .== (y .> 0.5))


# https://github.com/FluxML/Flux.jl/blob/0e36af98f6fc5b7f3c95fe819a02172cfaaaf777/src/gradient.jl#L3
# https://github.com/FluxML/Zygote.jl/blob/1b914d994aea236bcb6d3d0cd6c099d86cede101/src/compiler/interface.jl#L152


# Optimisers
# https://github.com/FluxML/Flux.jl/blob/0e36af98f6fc5b7f3c95fe819a02172cfaaaf777/src/layers/basic.jl#L85
# update! https://github.com/FluxML/Flux.jl/blob/0e36af98f6fc5b7f3c95fe819a02172cfaaaf777/src/optimise/train.jl


function mse_loss(ŷ::GraphNode, y::GraphNode)
    diff = ŷ .- y
    squared = diff .^ Constant(2)
    return Constant(0.5) .* squared
end

function cross_entropy_loss(ŷ::GraphNode, y::GraphNode)
    ϵ = Constant(eps(Float32))
    ŷ = ŷ .+ ϵ
    loss = Constant(-1.0f0) .* (y .* log.(ŷ))
    return sum(loss)
end

function gradient(model::Network)
    grads = []
    for layer in model.layers
        if isa(layer, Dense)
            push!(grads, layer.weights.∇)
            if !isnothing(layer.bias)
                push!(grads, layer.bias.∇)
            end
        end
    end
    return grads
end

function update_params!(model::Network, lr::Float32; grads::Any, batch_len::Integer)
    for (layer, ∇) in zip(model.layers, grads)
        if isa(layer, Dense)
            layer.weights.output .-= lr * ∇ / batch_len
            if !isnothing(layer.bias)
                layer.bias.output .-= lr * ∇ / batch_len
            end
        end
    end
    return nothing
end

# Initialize empty gradients with the same structure as model gradients
function init_zero_gradients(model::Network)
    grads = []
    for layer in model.layers
        if isa(layer, Dense) 
            # Create zero arrays with same shape as gradient
            push!(grads, zeros(eltype(layer.weights.∇), size(layer.weights.∇)))
            if !isnothing(layer.bias)
                push!(grads, zeros(eltype(layer.bias.∇), size(layer.bias.∇)))
            end
        end
    end
    return grads
end

# Accumulate gradients
function accumulate_gradients!(grads, layer_grads)
    for (g, lg) in zip(grads, layer_grads)
        g .+= lg  # Element-wise addition
    end
    return grads
end


linear(x) = x
ReLU(x) = max.(Constant(0.0), x)
swish(x) = x ./ (Constant(1.0) .+ exp.(Constant(-1) * x))