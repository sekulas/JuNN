using JuAD

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

mutable struct IndexOperator <: Operator
    inputs::Tuple{GraphNode, GraphNode}
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
    
    indices = idxs.output
    weights = W.output
    
    if ndims(indices) == 1
        # Single sequence: indices is (seq_length,)
        result = weights[:, indices]  # Result: (embed_dim, seq_length)
    else
        # Batch of sequences: indices is (seq_length, batch_size)
        seq_len, batch_size = size(indices)
        embed_dim = size(weights, 1)
        
        # Initialize output tensor
        result = zeros(Float32, embed_dim, seq_len, batch_size)
        
        # Fill embeddings for each sequence in the batch
        for b in 1:batch_size
            for t in 1:seq_len
                idx = indices[t, b]
                result[:, t, b] .= weights[:, idx]
            end
        end
    end
    
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
