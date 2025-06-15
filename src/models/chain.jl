struct Chain{T<:Tuple}
    layers::T
end

Chain(layers...) = Chain(layers)

(c::Chain)(x) = _apply_layer_sequence(c.layers, x)
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

function get_params(model::Chain)
    params = []
    i = 1
    for layer in model.layers
        if isa(layer, Dense)
            println("Adding Dense layer parameters: ", i)
            push!(params, layer.weights)
            if !isnothing(layer.bias)
                println("Adding Dense layer bias: ", i)
                push!(params, layer.bias)
            end
        elseif isa(layer, RNN)
            println("Adding RNN layer parameters: ", i)
            push!(params, layer.cell.W_ih)
            push!(params, layer.cell.W_hh)
            if !isnothing(layer.cell.bias)
                println("Adding RNN layer bias: ", i)
                push!(params, layer.cell.bias)
            end
        elseif isa(layer, Embedding)
            println("Adding Embedding layer parameters: ", i)
            push!(params, layer.weights)
        end
        i += 1
    end
    return params
end
