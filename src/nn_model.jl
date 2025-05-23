using Random: GLOBAL_RNG, AbstractRNG
include("structs.jl")
include("broadcast_operators.jl")
include("layers.jl")

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

function NeuralNetwork(model::Chain, optimizer::Any, loss::Function, accuracy::Function)
    input_size = size(model.layers[1].weights.output, 2)
    output_size = size(model.layers[end].weights.output, 1)
    
    x_node = Variable(zeros(Float32, input_size, 1), name="x_input")
    y_node = Variable(zeros(Float32, output_size, 1), name="y_true")
    
    y_pred_node = model(x_node)
    loss_node = loss(y_node, y_pred_node)
    sorted_graph = topological_sort(loss_node)

    params = get_params(model)
    
    NeuralNetwork(model, optimizer, loss, accuracy, x_node, y_node, y_pred_node, sorted_graph, params)
end

function train!(net::NeuralNetwork, dataset::DataLoader)
    batch_size = dataset.batchsize
    total_loss = 0.0f0
    total_acc = 0.0f0
    iterations = 0
    grads = [zeros(size(p.output)) for p in net.params]

    for (x_batch, y_batch) in dataset
        loss, acc = 
            gradient!(grads, net, x_batch, y_batch, batch_size)

        optimize!(net.optimizer, net.params, grads)
        
        total_loss += loss
        total_acc += acc

        iterations += 1
    end

    return (total_loss / iterations, total_acc / iterations)
end

function gradient!(grads, net, x_batch, y_batch, batch_size)
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

        backward!(net.sorted_graph)
        
        accumulate_gradients!(grads, net.params)
    end

    for i in 1:length(grads)
        grads[i] ./= batch_size
    end
    
    return (batch_loss / batch_size, batch_acc / batch_size) 
end

function accumulate_gradients!(grad_accumulator::Vector, params::Vector)
    for (i, param) in enumerate(params)
        grad_accumulator[i] .+= param.∇
    end
end

function evaluate(net, testset::DataLoader)
    batch_size = testset.batchsize
    total_loss = 0.0f0
    total_acc = 0.0f0
    iterations = 0

    for (x_batch, y_batch) in testset
        loss, acc = 
            test!(net, x_batch, y_batch, batch_size)
        
        total_loss += loss
        total_acc += acc

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