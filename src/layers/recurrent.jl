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
    
    function RNN(input_size::Int, hidden_size::Int;
                bias::Bool = true,
                activation = ReLU,
                init = glorot_uniform,
                name = nothing)
        
        cell = RNNCell(input_size, hidden_size; 
                      bias=bias, activation=activation, init=init, name=name)
        new(cell, hidden_size)
    end
end

function (rnn::RNN)(x::GraphNode, initial_hidden::Union{GraphNode, Nothing} = nothing)   
    input_dims = size(x.output)
    _, seq_length, batch_size = input_dims
    
    if initial_hidden === nothing
        hidden = Variable(zeros(Float32, rnn.hidden_size, batch_size), name="h0")
    else
        hidden = initial_hidden
    end
    
    for t in 1:seq_length
        x_t = getindex_col_batch(x, Constant(t))
        x_t.output = JuAD.forward(x_t, x.output, t)
        
        hidden = rnn.cell(x_t, hidden)
    end
    
    return hidden
end