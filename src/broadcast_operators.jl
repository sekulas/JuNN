import Base: *
import LinearAlgebra: mul!
# x * y (aka matrix multiplication)
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)



# x .* y (element-wise multiplication)
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) =
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(y .* 𝟏)
        Jy = diagm(x .* 𝟏)
        tuple(Jx' * g, Jy' * g)
    end



Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = 
    BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = x.- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)



Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = 
    BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)



import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = 
    let
        𝟏 = ones(length(x))
        J = 𝟏'
        tuple(J' * g)
    end



Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) =
    BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) = 
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(𝟏 ./ y)
        Jy = (-x ./ y .^2)
        tuple(Jx' * g, Jy' * g)
    end



import Base: max
Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = 
    BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = 
    let
        Jx = diagm(isless.(y, x))
        Jy = diagm(isless.(x, y))
        tuple(Jx' * g, Jy' * g)
    end



σ(x) = BroadcastedOperator(σ, x)
forward(::BroadcastedOperator{typeof(σ)}, x) = 1.0 ./ (1.0 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(σ)}, x, g) = 
    let
        y = node.output
        𝟏 = ones(length(y))
        J = diagm(y .* (1.0 .- y))
        tuple(J' * g)
    end

Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = 
    BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x, y) = 
    x .^ y
backward(node::BroadcastedOperator{typeof(^)}, x, y, g) = 
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(y .* x .^ (y .- 1.0))
        Jy = diagm(log.(abs.(x)) .* x .^ y)
        tuple(Jx' * g, Jy' * g)
    end

Base.Broadcast.broadcasted(exp, x::GraphNode) = 
    BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x) = 
    exp.(x)
backward(node::BroadcastedOperator{typeof(exp)}, x, g) = 
    let
        y = node.output
        J = diagm(y)
        tuple(J' * g)
    end

Base.Broadcast.broadcasted(log, x::GraphNode) = 
    BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = 
    log.(x)
backward(::BroadcastedOperator{typeof(log)}, x, g) = 
    tuple(diagm(1.0 ./ x)' * g)


softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) = exp.(x) ./ sum(exp.(x))
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) = 
    let
        y = node.output
        J = diagm(y) .- y * y'
        tuple(J' * g)
    end