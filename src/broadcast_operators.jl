import Base: *
import LinearAlgebra: mul!
# x * y (aka matrix multiplication)
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, ∇) = tuple(∇ * x', A' * ∇)


import LinearAlgebra: diagm
# x .* y (element-wise multiplication)
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, ∇) =
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(y .* 𝟏)
        Jy = diagm(x .* 𝟏)
        tuple(Jx' * ∇, Jy' * ∇)
    end
# ( ∇ .* y,  ∇ .* x )

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = 
    BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = x.- y
backward(::BroadcastedOperator{typeof(-)}, x, y, ∇) = tuple(∇, -∇)



Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = 
    BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, ∇) = tuple(∇, ∇)



import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, ∇) = 
    let
        𝟏 = ones(length(x))
        J = 𝟏'
        tuple(J' * ∇)
    end
# ( fill(∇, size(x)), )


# Potencjalnie do potegi ^-1 zamiast dzielenia w funkcji aktywacji.
Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) =
    BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, ∇) = 
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(𝟏 ./ y)
        Jy = (-x ./ y .^2)
        tuple(Jx' * ∇, Jy' * ∇)
    end
    # ( ∇ ./ y,
    #  -∇ .* x ./ (y .^ 2) )


import Base: max
Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = 
    BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, ∇) = 
    let
        Jx = diagm(isless.(y, x))
        Jy = diagm(isless.(x, y))
        tuple(Jx' * ∇, Jy' * ∇)
    end
    # let mx = x .> y      # or `>=` to break ties differently
    #     in ( ∇ .* mx, ∇ .* .!mx )


σ(x) = BroadcastedOperator(σ, x)
forward(::BroadcastedOperator{typeof(σ)}, x) = 1.0 ./ (1.0 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(σ)}, x, ∇) = 
    # let
    #     y = node.output
    #     𝟏 = ones(length(y))
    #     J = diagm(y .* (1.0 .- y))
    #     tuple(J' * ∇)
    # end
    let 
        y = node.output
        tuple(∇ .* (y .* (1 .- y)))
    end

Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = 
    BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x, y) = 
    x .^ y
backward(node::BroadcastedOperator{typeof(^)}, x, y, ∇) = 
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(y .* x .^ (y .- 1.0))
        Jy = diagm(log.(abs.(x)) .* x .^ y)
        tuple(Jx' * ∇, Jy' * ∇)
    end
    # ( ∇ .* (y .* x .^ (y .- 1)),
    #   ∇ .* (log.(abs.(x)) .* x .^ y) )

Base.Broadcast.broadcasted(exp, x::GraphNode) = 
    BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x) = 
    exp.(x)
backward(node::BroadcastedOperator{typeof(exp)}, x, ∇) = 
    let
        y = node.output
        J = diagm(y)
        tuple(J' * ∇)
    end
    # ( ∇ .* node.output, )

Base.Broadcast.broadcasted(log, x::GraphNode) = 
    BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = 
    log.(x)
backward(::BroadcastedOperator{typeof(log)}, x, ∇) = 
    tuple(diagm(1.0 ./ x)' * ∇)
    # ( ∇ ./ x, )


softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) = exp.(x) ./ sum(exp.(x))
backward(node::BroadcastedOperator{typeof(softmax)}, x, ∇) = 
    # let
    #     y = node.output
    #     J = diagm(y) .- y * y'
    #     tuple(J' * ∇)
    # end
    let
        y = node.output
        ω = sum(∇ .* y)
        tuple(y .* (∇ .- ω))
    end

Base.Broadcast.broadcasted(identity, x::GraphNode) = BroadcastedOperator(identity, x)
forward(::BroadcastedOperator{typeof(identity)}, x) = x
backward(::BroadcastedOperator{typeof(identity)}, x, ∇) = 
    tuple(∇)


import Base: tanh
tanh(x::GraphNode) = BroadcastedOperator(tanh, x)
forward(::BroadcastedOperator{typeof(tanh)}, x) = tanh.(x)
backward(node::BroadcastedOperator{typeof(tanh)}, x, ∇) =
  let
    y = node.output
    tuple((1 .- y .^ 2) .* ∇)
  end