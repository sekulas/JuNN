function mse_loss(y::GraphNode, ŷ::GraphNode)
    diff = ŷ .- y
    squared = diff .^ Constant(2)
    return Constant(0.5) .* squared
end

function cross_entropy_loss(y::GraphNode, ŷ::GraphNode)
    ϵ = Constant(eps(Float32))
    ŷ = ŷ .+ ϵ
    loss = Constant(-1.0f0) .* (y .* log.(ŷ))
    return sum(loss)
end

function binary_cross_entropy(y::GraphNode, ŷ::GraphNode)
    ϵ = Constant(eps(Float32))
    losses = Constant(-1.0f0) .* y .* log.(ŷ .+ ϵ) .- (Constant(1.0f0) .- y) .* log.(Constant(1.0f0) .- ŷ .+ ϵ)
    return mean(losses)
end