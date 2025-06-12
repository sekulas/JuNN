using JuAD

mutable struct Dense{F}
    weights::Variable
    bias::Union{Nothing, Variable}
    activation::F

    function Dense(weights::Variable, 
                   bias::Bool = true, 
                   activation::F = identity) where {F}
        b = create_bias(weights, bias)
        new{F}(weights, b, activation)
    end
end

function create_bias(weights::Variable, bias::Bool)
    bias ? Variable(fill!(similar(weights.output, size(weights.output,1)), 0), 
                    name="$(weights.name)_bias") : nothing
end

function Dense((in, out)::Pair{<:Integer, <:Integer}, activation::F=identity;
               init = glorot_uniform, 
               bias::Bool = true, 
               name=nothing) where {F}
    name = isnothing(name) ? "dense_$in=>$out" : name
    Dense(Variable(init(out, in), name=name), bias, activation)
end

function (l::Dense)(x::GraphNode)
    z = l.weights * x
    
    if l.bias !== nothing
        z = z .+ l.bias
    end
    
    return l.activation(z)
end