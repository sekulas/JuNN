update!(node::Constant, ∇) = nothing
update!(node::GraphNode, ∇) = 
    if isnothing(node.∇)
        node.∇ = ∇ 
    else
        node.∇ .+= ∇
    end

function backward!(order::Vector; seed=1.0f0)
    result = last(order)
    result.∇ = seed
    @assert length(result.output) == 1 "∇ is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.∇)
    for (input, ∇) in zip(inputs, gradients)
        update!(input, ∇)
    end
    return nothing
end