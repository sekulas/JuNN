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

# Create the IndexOperator and immediately compute its output for graph construction
# (downstream layers can access size information)
function (l::Embedding)(indices::GraphNode)
    op = IndexOperator(l.weights, indices; name="embedding_output")
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

function JuAD.compute!(node::IndexOperator)
    node.output = forward(node, node.inputs[1], node.inputs[2])
end

# W.output - embedding weight matrix (embed_dim × vocab_size)
# idxs.output - integer indices (seq_length, batch_size)
function JuAD.forward(op::IndexOperator, W::Variable, idxs::GraphNode)
    indices = idxs.output
    weights = W.output
    
    seq_len, batch_size = size(indices)
    embed_dim = size(weights, 1)
    
    result = zeros(Float32, embed_dim, seq_len, batch_size)
    
    for b in 1:batch_size
        for t in 1:seq_len
            idx = indices[t, b]
            result[:, t, b] .= weights[:, idx]
        end
    end
    
    return result
end


# ∇_output shape as op.output
# W.output (embed_dim, vocab_size)
function JuAD.backward(op::IndexOperator, W, idxs, ∇_output::AbstractArray)
    indices = idxs
    embed_dim, vocab_size = size(W)
    
    ∇_W = zeros(Float32, embed_dim, vocab_size)
    
    seq_len, batch_size = size(indices)
    for b in 1:batch_size
        for t in 1:seq_len
            idx = indices[t, b]
            ∇_W[:, idx] .+= ∇_output[:, t, b]
        end
    end
    
    return (∇_W, nothing)
end
