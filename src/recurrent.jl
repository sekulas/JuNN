# mutable struct Embedding
#     weights::Variable
    
#     function Embedding(vocab_size::Int, embed_dim::Int; 
#                       init=glorot_uniform, 
#                       name=nothing)
#         name = isnothing(name) ? "embedding_$(vocab_size)x$(embed_dim)" : name
#         weights = Variable(init(embed_dim, vocab_size), name=name)
#         new(weights)
#     end
# end

# function (l::Embedding)(indices::GraphNode)
#     op = IndexOperator(l.weights, indices; name="embedding_output")

#     forward(op, l.weights, indices)

#     op
# end

# # RNN Cell - implements the core RNN computation
# mutable struct RNNCell
#     W_ih::Variable  # Input to hidden weights
#     W_hh::Variable  # Hidden to hidden weights  
#     bias::Union{Nothing, Variable}
#     activation::Function
    
#     function RNNCell(input_size::Int, hidden_size::Int;
#                     bias::Bool = true,
#                     activation = tanh,
#                     init = glorot_uniform,
#                     name = nothing)
        
#         name_prefix = isnothing(name) ? "rnn_cell" : name
        
#         W_ih = Variable(init(hidden_size, input_size), name="$(name_prefix)_W_ih")
#         W_hh = Variable(init(hidden_size, hidden_size), name="$(name_prefix)_W_hh")
        
#         b = bias ? Variable(zeros(Float32, hidden_size, 1), name="$(name_prefix)_bias") : nothing
        
#         new(W_ih, W_hh, b, activation)
#     end
# end

# function (cell::RNNCell)(input::GraphNode, hidden::GraphNode)
#     # RNN cell computation: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
#     ih = cell.W_ih * input
#     hh = cell.W_hh * hidden
    
#     combined = ih .+ hh
    
#     if cell.bias !== nothing
#         combined = combined .+ cell.bias
#     end
    
#     return cell.activation(combined)
# end

# # RNN Layer - processes sequences
# mutable struct RNN
#     cell::RNNCell
#     hidden_size::Int
#     return_sequences::Bool
    
#     function RNN(input_size::Int, hidden_size::Int;
#                 bias::Bool = true,
#                 activation = tanh,
#                 return_sequences::Bool = false,
#                 init = glorot_uniform,
#                 name = nothing)
        
#         cell = RNNCell(input_size, hidden_size; 
#                       bias=bias, activation=activation, init=init, name=name)
#         new(cell, hidden_size, return_sequences)
#     end
# end

# function (rnn::RNN)(x::GraphNode, initial_hidden::Union{GraphNode, Nothing} = nothing)
#     # x should be of shape (input_size, seq_length, batch_size) or (input_size, seq_length) for single batch
#     #println("RNN forward with input shape: ", x)
#     input_dims = size(x.output)
    
#     if length(input_dims) == 2
#         input_size, seq_length = input_dims
#         batch_size = 1
#     else
#         input_size, seq_length, batch_size = input_dims
#     end
    
#     # Initialize hidden state
#     if initial_hidden === nothing
#         hidden = Variable(zeros(Float32, rnn.hidden_size, batch_size), name="h0")
#     else
#         hidden = initial_hidden
#     end
    
#     outputs = []
    
#     # Process sequence step by step
#     for t in 1:seq_length
#         if length(input_dims) == 2
#             x_t = Variable(x.output[:, t:t], name="x_$t")
#         else
#             x_t = Variable(x.output[:, t, :], name="x_$t")
#         end
        
#         hidden = rnn.cell(x_t, hidden)
        
#         if rnn.return_sequences
#             push!(outputs, hidden)
#         end
#     end
    
#     if rnn.return_sequences
#         # Concatenate all outputs
#         # This is simplified - you'd want a proper concatenation operator
#         return outputs[end]  # For now, return last output
#     else
#         return hidden
#     end
# end

# # Reset function for RNN (needed for training)
# function reset!(rnn::RNN)
#     # In stateful RNNs, you might want to reset hidden states
#     # For now, this is a placeholder since we create new hidden states each forward pass
#     nothing
# end

# Updated Chain to handle RNN reset
function reset!(model::Chain)
    for layer in model.layers
        if isa(layer, RNN)
            reset!(layer)
        end
    end
end

# mutable struct IndexOperator <: Operator
#     inputs::Tuple{GraphNode, GraphNode}  # (weights, indices)
#     output :: Any
#     ∇      :: Any
#     name   :: String

#     function IndexOperator(weights::GraphNode, indices::GraphNode; name="index")
#         new((weights, indices), nothing, nothing, name)
#     end
# end


# function forward(op::IndexOperator, W::GraphNode, idxs::GraphNode)
#     # W is the embedding‐weight matrix (size = embed_dim × vocab_size)
#     # idxs is “indices” (an integer vector of length seq_length)
#     # We select the corresponding columns of W, producing a (embed_dim × seq_length) array:
#     op.output = @view(W.output[:, idxs.output])           # or `W[:, idxs]` if you prefer a copy
# end

# function backward(op::IndexOperator, W::GraphNode, idxs::GraphNode, ∂Y::AbstractMatrix{<:Number})
#     # ∂Y is the gradient wrt. the embedding outputs (shape = embed_dim × seq_length).
#     # We need to scatter‐accumulate ∂Y into a gradient the same shape as W
#     gradW = zero(W)                          # same size as W (embed_dim × vocab_size)
#     for (t, token) in enumerate(idxs)
#         @views gradW[:, token] .+= ∂Y[:, t]
#     end
#     return (gradW, nothing)   # no gradient w.r.t. the indices vector
# end

# Fixed recurrent.jl with proper IndexOperator gradient handling

mutable struct Embedding
    weights::Variable
    
    function Embedding(vocab_size::Int, embed_dim::Int; 
                      init=glorot_uniform, 
                      name=nothing)
        name = isnothing(name) ? "embedding_$(vocab_size)x$(embed_dim)" : name
        weights = Variable(init(embed_dim, vocab_size), name=name)
        new(weights)
    end
end

function (l::Embedding)(indices::GraphNode)
    # Create the IndexOperator and immediately compute its output for graph construction
    op = IndexOperator(l.weights, indices; name="embedding_output")
    # Immediately compute the output so downstream layers can access size information
    op.output = forward(op, l.weights, indices)
    return op
end

# RNN Cell - implements the core RNN computation
mutable struct RNNCell
    W_ih::Variable  # Input to hidden weights
    W_hh::Variable  # Hidden to hidden weights  
    bias::Union{Nothing, Variable}
    activation::Function
    
    function RNNCell(input_size::Int, hidden_size::Int;
                    bias::Bool = true,
                    activation = tanh,
                    init = glorot_uniform,
                    name = nothing)
        
        name_prefix = isnothing(name) ? "rnn_cell" : name
        
        W_ih = Variable(init(hidden_size, input_size), name="$(name_prefix)_W_ih")
        W_hh = Variable(init(hidden_size, hidden_size), name="$(name_prefix)_W_hh")
        
        b = bias ? Variable(zeros(Float32, hidden_size, 1), name="$(name_prefix)_bias") : nothing
        
        new(W_ih, W_hh, b, activation)
    end
end

function (cell::RNNCell)(input::GraphNode, hidden::GraphNode)
    # RNN cell computation: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
    ih = cell.W_ih * input
    hh = cell.W_hh * hidden
    
    combined = ih .+ hh
    
    if cell.bias !== nothing
        combined = combined .+ cell.bias
    end
    
    return cell.activation(combined)
end

# RNN Layer - processes sequences
mutable struct RNN
    cell::RNNCell
    hidden_size::Int
    return_sequences::Bool
    
    function RNN(input_size::Int, hidden_size::Int;
                bias::Bool = true,
                activation = tanh,
                return_sequences::Bool = false,
                init = glorot_uniform,
                name = nothing)
        
        cell = RNNCell(input_size, hidden_size; 
                      bias=bias, activation=activation, init=init, name=name)
        new(cell, hidden_size, return_sequences)
    end
end

function (rnn::RNN)(x::GraphNode, initial_hidden::Union{GraphNode, Nothing} = nothing)
    # x should be of shape (embed_dim, seq_length) after embedding
    # Handle the case where x.output might be nothing during graph construction
    if x.output === nothing
        error("RNN input has not been computed. Make sure the embedding layer computes its output during construction.")
    end
    
    input_dims = size(x.output)
    
    if length(input_dims) == 2
        input_size, seq_length = input_dims
        batch_size = 1
    else
        input_size, seq_length, batch_size = input_dims
    end
    
    # Initialize hidden state
    if initial_hidden === nothing
        hidden = Variable(zeros(Float32, rnn.hidden_size, batch_size), name="h0")
    else
        hidden = initial_hidden
    end
    
    outputs = []
    
    # Process sequence step by step
    for t in 1:seq_length
        if length(input_dims) == 2
            # Extract the t-th column (timestep) from the embedded sequence
            x_t = BroadcastedOperator(getindex_col, x, Constant(t))
            # Compute the output immediately for graph construction
            x_t.output = forward(x_t, x, Constant(t))
        else
            x_t = BroadcastedOperator(getindex_col_batch, x, Constant(t))
            # Compute the output immediately for graph construction
            x_t.output = forward(x_t, x, Constant(t))
        end
        
        hidden = rnn.cell(x_t, hidden)
        
        if rnn.return_sequences
            push!(outputs, hidden)
        end
    end
    
    if rnn.return_sequences
        return outputs[end]  # For now, return last output
    else
        return hidden
    end
end

# Helper functions for extracting columns (timesteps) from embedded sequences
getindex_col(x, t) = @view x[:, t:t]
getindex_col_batch(x, t) = @view x[:, t, :]

# Forward and backward for column extraction
forward(::BroadcastedOperator{typeof(getindex_col)}, x, t) = @view x.output[:, t.output:t.output]
backward(::BroadcastedOperator{typeof(getindex_col)}, x, t, ∇) = 
    let
        grad_x = zeros(eltype(∇), size(x.output))
        grad_x[:, t.output:t.output] .= ∇
        (grad_x, nothing)
    end

forward(::BroadcastedOperator{typeof(getindex_col_batch)}, x::Array{Float32}, t::Int64) = @view x[:, t:t]
forward(::BroadcastedOperator{typeof(getindex_col_batch)}, x, t) = @view x.output[:, t.output, :]
backward(::BroadcastedOperator{typeof(getindex_col_batch)}, x::Array{Float32}, t::Int64, ∇) = 
    let
        grad_x = zeros(eltype(∇), size(x))
        grad_x[:, t:t] .= ∇
        (grad_x, nothing)
    end
# backward(::BroadcastedOperator{typeof(getindex_col_batch)}, x, t, ∇) = 
#     let
#         grad_x = zeros(eltype(∇), size(x.output))
#         grad_x[:, t.output, :] .= ∇
#         (grad_x, nothing)
#     end

# Reset function for RNN
function reset!(rnn::RNN)
    nothing
end

# FIXED IndexOperator - This is the key fix for your gradient issue
mutable struct IndexOperator <: Operator
    inputs::Tuple{GraphNode, GraphNode}  # (weights, indices)
    output :: Any
    ∇      :: Any
    name   :: String

    function IndexOperator(weights::GraphNode, indices::GraphNode; name="index")
        new((weights, indices), nothing, nothing, name)
    end
end

# Add compute! method for IndexOperator to integrate with your forward! system
function compute!(node::IndexOperator)
    node.output = forward(node, node.inputs[1], node.inputs[2])
end

# Forward pass: extract embeddings for given indices
function forward(op::IndexOperator, W::Variable, idxs::GraphNode)
    # W.output is the embedding weight matrix (embed_dim × vocab_size)
    # idxs.output contains integer indices (seq_length,) or (seq_length, batch_size)
    
    # indices = idxs.output
    # weights = W.output
    
    # if ndims(indices) == 1
        # Single sequence: indices is (seq_length,)
    result = @view W.output[:, idxs.output]  # Result: (embed_dim, seq_length)
    # else
    #     # Batch of sequences: indices is (seq_length, batch_size)
    #     seq_len, batch_size = size(indices)
    #     embed_dim = size(weights, 1)
        
    #     # Initialize output tensor
    #     result = zeros(Float32, embed_dim, seq_len, batch_size)
        
    #     # Fill embeddings for each sequence in the batch
    #     for b in 1:batch_size
    #         for t in 1:seq_len
    #             idx = indices[t, b]
    #             result[:, t, b] .= weights[:, idx]
    #         end
    #     end
    # end
    
    return result
end

# CRITICAL FIX: Proper backward pass for IndexOperator
function backward(op::IndexOperator, W, idxs, ∇_output::AbstractArray)
    # ∇_output has the same shape as op.output
    # W.output is (embed_dim, vocab_size)
    # We need to accumulate gradients back to the embedding weights
    
    indices = idxs
    embed_dim, vocab_size = size(W)
    
    # Initialize gradient for weights (same size as W.output)
    ∇_W = zeros(Float32, embed_dim, vocab_size)
    
    if ndims(indices) == 1
        # Single sequence case
        seq_len = length(indices)
        for t in 1:seq_len
            idx = indices[t]
            # Accumulate gradient for this embedding vector
            ∇_W[:, idx] .+= ∇_output[:, t]
        end
    else
        # Batch case
        seq_len, batch_size = size(indices)
        for b in 1:batch_size
            for t in 1:seq_len
                idx = indices[t, b]
                # Accumulate gradient for this embedding vector
                ∇_W[:, idx] .+= ∇_output[:, t, b]
            end
        end
    end
    
    # Return gradients: (gradient w.r.t. weights, gradient w.r.t. indices)
    # No gradient w.r.t. indices (they're discrete)
    return (∇_W, nothing)
end

