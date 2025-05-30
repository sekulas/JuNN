import Base: *
import LinearAlgebra: mul!, transpose!
# x * y (aka matrix multiplication)
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = A * x
backward(node::BroadcastedOperator{typeof(mul!)}, A, x, ∇) = let 
    if isnothing(node.buffers)
        A_size = size(A)
        x_size = size(x)
        
        node.buffers = (
            dA = zeros(eltype(A), A_size...),
            dx = zeros(eltype(x), x_size...),
            temp_x_t = zeros(eltype(x), reverse(x_size)...)
        )
    end

    dA_buf = node.buffers.dA
    dx_buf = node.buffers.dx
    temp_x_t = node.buffers.temp_x_t

    transpose!(temp_x_t, x)
    mul!(dA_buf, ∇, temp_x_t) #∇ * x'
    mul!(dx_buf, A', ∇) #A' * ∇

    return (dA_buf, dx_buf)
end 


import LinearAlgebra: diagm
# x .* y (element-wise multiplication)
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, ∇) =
    # let
    #     𝟏 = ones(length(node.output))
    #     Jx = diagm(y .* 𝟏)
    #     Jy = diagm(x .* 𝟏)
    #     tuple(Jx' * ∇, Jy' * ∇)
    # end
    ( ∇ .* y,  ∇ .* x )

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
    # let
    #     𝟏 = ones(length(x))
    #     J = 𝟏'
    #     tuple(J' * ∇)
    # end
    ( fill(∇, size(x)), )

import Statistics: mean
mean(x::GraphNode) = BroadcastedOperator(mean, x)
forward(::BroadcastedOperator{typeof(mean)}, x) =
    mean(x)
backward(::BroadcastedOperator{typeof(mean)}, x, ∇) =
    let n = length(x)
        δ = fill(∇ / n, size(x))
    in
        tuple(δ)
    end



# Potencjalnie do potegi ^-1 zamiast dzielenia w funkcji aktywacji.
Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) =
    BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y, ∇) = 
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


sigmoid(x::GraphNode) = BroadcastedOperator(σ, x)
σ(x::GraphNode) = BroadcastedOperator(σ, x)
forward(::BroadcastedOperator{typeof(σ)}, x) = 1.0f0 ./ (1.0f0 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(σ)}, x, ∇) = 
    # let
    #     y = node.output
    #     𝟏 = ones(length(y))
    #     J = diagm(y .* (1.0 .- y))
    #     tuple(J' * ∇)
    # end
    let 
        y = node.output
        tuple(∇ .* (y .* (1.0f0 .- y)))
    end

Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = 
    BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x, y) = 
    x .^ y
backward(node::BroadcastedOperator{typeof(^)}, x, y, ∇) = 
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(y .* x .^ (y .- 1.0f0))
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
    # tuple(diagm(1.0 ./ x)' * ∇)
    ( ∇ ./ x, )


softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) =
    let 
        max_x = maximum(x)
        exp_x_shifted = exp.(x .- max_x)
        sum_exp_x_shifted = sum(exp_x_shifted)
        y = exp_x_shifted ./ sum_exp_x_shifted 
    end
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
    tuple((1.0f0 .- y .^ 2.0f0) .* ∇)
  end

ReLU(x::GraphNode) = BroadcastedOperator(ReLU, x)
forward(::BroadcastedOperator{typeof(ReLU)}, x) = max.(0.0f0, x)
backward(node::BroadcastedOperator{typeof(ReLU)}, x, ∇) =
    let
        y    = node.output
        mask = y .> 0.0f0
        tuple(mask .* ∇)            
    end