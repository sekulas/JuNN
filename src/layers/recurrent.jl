using JuAD

mutable struct RNNCell
    W_ih::Variable  
    W_hh::Variable  
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
    ih = cell.W_ih * input
    hh = cell.W_hh * hidden
    
    combined = ih .+ hh
    
    if cell.bias !== nothing
        combined = combined .+ cell.bias
    end
    
    return cell.activation(combined)
end

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
            # x_t = BroadcastedOperator(getindex_col, x, Constant(t))
            x_t = getindex_col(x, Constant(t))
            # Compute the output immediately for graph construction
            x_t.output = forward(x_t, x.output, t)
        else
            x_t = getindex_col_batch(x, Constant(t))
            # Compute the output immediately for graph construction
            x_t.output = forward(x_t, x.output, t)
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