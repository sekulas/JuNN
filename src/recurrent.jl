const BATCH_SIZE = 1;

# Updated Chain to handle RNN reset
function reset!(model::Chain)
    for layer in model.layers
        if isa(layer, RNN)
            reset!(layer)
        end
    end
end

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
                    activation = ReLU,
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
                activation = ReLU,
                return_sequences::Bool = false,
                init = glorot_uniform,
                name = nothing)
        
        cell = RNNCell(input_size, hidden_size; 
                      bias=bias, activation=activation, init=init, name=name)
        new(cell, hidden_size, return_sequences)
    end
end

function (rnn::RNN)(x::GraphNode, initial_hidden::Union{GraphNode, Nothing} = nothing)
    input_dims = size(x.output)
    _, seq_length = input_dims
    
    # Initialize hidden state
    if initial_hidden === nothing
        hidden = Variable(zeros(Float32, rnn.hidden_size, BATCH_SIZE), name="h0")
    else
        hidden = initial_hidden
    end
    
    outputs = []
    
    # Process sequence step by step
    for t in 1:seq_length
        # Extract the t-th column (timestep) from the embedded sequence
        # x_t = BroadcastedOperator(getindex_col, x, Constant(t))
        x_t = getindex_col(x, Constant(t))
        # Compute the output immediately for graph construction
        x_t.output = forward(x_t, x.output, t)

        
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
    indices = idxs.output
    weights = W.output
   
    result = weights[:, @view(indices[:, 1])]
   
    return result
end

# CRITICAL FIX: Proper backward pass for IndexOperator
function backward(op::IndexOperator, W, idxs, ∇_output::AbstractArray)
    embed_dim, vocab_size = size(W)
    
    # TODO: NOT NECESSARY ALLOCATIONS
    ∇_W = zeros(Float32, embed_dim, vocab_size)
    
    N = size(idxs, 1)
    
    for i in 1:N
        idx = idxs[i, 1]
        ∇_W[:, idx] .+= ∇_output[:, i]
    end
    
    return (∇_W, nothing)
end

