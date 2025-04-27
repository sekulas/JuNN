abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    ∇ :: Any
    name :: String
    Variable(output, ∇, name) = new(output, ∇, name)
end
Variable(output; name="?") = Variable(output, nothing, name)

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    ∇ :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    ∇ :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end