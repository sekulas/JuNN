using JuAD
include("../data_loader.jl")

struct NeuralNetwork
    model::Chain
    optimizer::Any
    loss::Function
    accuracy::Function
    x_node::Variable
    y_node::Variable
    y_pred_node::GraphNode
    sorted_graph::Vector{GraphNode}
    params::Vector{Variable}
end

function NeuralNetwork(model::Chain, optimizer::Any, loss::Function, accuracy::Function, batch_size::Int64; seq_length::Union{Int, Nothing}=nothing)
    input_size = size(model.layers[1].weights.output, 2)
    output_size = size(model.layers[end].weights.output, 1)

    data_type = Float32
    init_value = zeros
    
    if model.layers[1] isa Embedding
        if seq_length !== nothing
            input_size = seq_length
        else
            println("Warning: seq_length is not provided, using input_size from the embedding layer.")
            println("Solution may not work as expected if seq_length is not set.")
        end
        data_type = Int32
        init_value = ones
    end
    
    x_node = Variable(init_value(data_type, input_size, batch_size), name="x_input")
    y_node = Variable(zeros(Float32, output_size, batch_size), name="y_true")
    
    y_pred_node = model(x_node)
    loss_node = loss(y_node, y_pred_node)
    sorted_graph = topological_sort(loss_node)

    params = get_params(model)
    
    NeuralNetwork(model, optimizer, loss, accuracy, x_node, y_node, y_pred_node, sorted_graph, params)
end

function train!(net::NeuralNetwork, dataset::DataLoader)
    total_loss = 0.0f0
    total_acc  = 0.0f0
    iterations = 0


    for (x_batch, y_batch) in dataset
        net.x_node.output = x_batch
        net.y_node.output = y_batch

        batch_loss = forward!(net.sorted_graph)
        batch_acc  = net.accuracy(y_batch, net.y_pred_node.output)

        backward!(net.sorted_graph)

        N = size(x_batch, 2)
        for param in net.params
            grad = param.∇


            if size(param.output, 2) == 1 && size(grad, 2) != 1
                grad = sum(grad; dims=2)    # now (out,1)
            end

            grad ./= float(N)               # average over minibatch
            apply!(net.optimizer, param, grad)
        end

        total_loss += batch_loss
        total_acc  += batch_acc
        iterations += 1
    end


    return (total_loss / iterations, total_acc / iterations)
end

function gradient!(grads, net, x_batch, y_batch, batch_size)
    batch_loss = 0.0f0
    batch_acc = 0.0f0

    batch_size = size(x_batch, 2)

    for i in 1:batch_size
        x_sample = @view x_batch[:, i:i]
        y_sample = @view y_batch[:, i:i]
        
        net.x_node.output .= x_sample
        net.y_node.output .= y_sample
        
        batch_loss += forward!(net.sorted_graph)
        batch_acc += net.accuracy(y_sample, net.y_pred_node.output)        

        backward!(net.sorted_graph)
        
        accumulate_gradients!(grads, net.params)
    end

    inv_batch_size = 1.0f0 / batch_size
    for grad in grads
        grad .*= inv_batch_size
    end
    
    return (batch_loss / batch_size, batch_acc / batch_size) 
end

function accumulate_gradients!(grad_accumulator::Vector, params::Vector)
    for (i, param) in enumerate(params)
        grad_accumulator[i] .+= param.∇
    end
end

function evaluate(net::NeuralNetwork, testset::DataLoader)
    total_loss = 0.0f0
    total_acc  = 0.0f0
    iterations = 0

    for (x_batch, y_batch) in testset
        net.x_node.output = x_batch
        net.y_node.output = y_batch

        batch_loss = forward!(net.sorted_graph)
        batch_acc  = net.accuracy(y_batch, net.y_pred_node.output)

        total_loss += batch_loss
        total_acc  += batch_acc
        iterations += 1
    end

    return (total_loss / iterations, total_acc / iterations)
end

function test!(net, x_batch, y_batch, batch_size)
    batch_loss = 0.0f0
    batch_acc = 0.0f0

    batch_size = size(x_batch, 2)

    for i in 1:batch_size
        x_sample = x_batch[:, i:i]
        y_sample = y_batch[:, i:i]
        
        net.x_node.output .= x_sample
        net.y_node.output .= y_sample
        
        batch_loss += forward!(net.sorted_graph)
        batch_acc += net.accuracy(y_sample, net.y_pred_node.output)        
    end

    return (batch_loss / batch_size, batch_acc / batch_size) 
end