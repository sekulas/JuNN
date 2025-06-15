import Base:^
^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = x^n
backward(::ScalarOperator{typeof(^)}, x, n, ∇) = 
    let
        ∇ * n * x^(n-1.0f0),
        ∇ * log(abs(x)) * x^n
    end


    
import Base: sin
sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = sin(x)
backward(::ScalarOperator{typeof(sin)}, x, ∇) = tuple(∇ * cos(x))