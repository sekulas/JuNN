reset!(node::Constant) = nothing
reset!(node::Variable) = node.∇ = nothing
reset!(node::Operator) = node.∇ = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end