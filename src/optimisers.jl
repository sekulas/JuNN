# https://github.com/FluxML/Flux.jl/blob/0e36af98f6fc5b7f3c95fe819a02172cfaaaf777/src/train.jl#L20
abstract type AbstractOptimiser end
const EPS = eps(Float32)

function update!(opt::AbstractOptimiser, x::AbstractArray, x̄)
  x̄r = copyto!(similar(x̄), x̄)  # Flux.Optimise assumes it can mutate the gradient. This is not
                               # safe due to aliasing, nor guaranteed to be possible, e.g. Fill.
  x .-= apply!(opt, x, x̄r)
end

mutable struct Adam <: AbstractOptimiser
    eta::Float32
    beta::Tuple{Float32,Float32}
    epsilon::Float32
    state::IdDict{Any, Any} #TODO: Change type?
end

"""
Construct an Adam optimizer.
Arguments:
- `η` (eta): Learning rate.
- `β` (beta): Tuple of decay rates for moment estimates (β1, β2).
- `ϵ` (epsilon): Small constant for numerical stability.
"""
Adam(η::Float32 = 0.001f0, β::Tuple{Float32,Float32} = (0.9f0, 0.999f0), ϵ::Float32 = EPS) = Adam(η, β, ϵ, IdDict())

# Internal constructor if state is pre-populated (e.g. for deserialization)
Adam(η::Float32, β::Tuple{Float32,Float32}, state::IdDict) = Adam(η, β, EPS, state)
  

"""
    apply!(o::Adam, x::AbstractArray, Δ::AbstractArray)

Compute the update step for parameters `x` with gradient `Δ` using Adam.
Note: This function modifies `Δ` in-place to store the result.
The input `Δ` should be a mutable copy of the original gradient if the original needs to be preserved.
"""
function apply!(o::Adam, x, Δ)
  η, β = o.eta, o.beta

  mt, vt, βp = get!(o.state, x) do
      (zero(x), zero(x), Float32[β[1], β[2]])
  end :: Tuple{typeof(x),typeof(x),Vector{Float32}}

  # Update biased first moment estimate
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  
  # Update biased second raw moment estimate
  @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)

  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon) * η
  βp .= βp .* β

  return Δ
end
  


function update_params!(model::Network, opt::Adam)
    for layer in model.layers
        if isa(layer, Dense)
            # Update weights
            if isdefined(layer.weights, :∇) && !isnothing(layer.weights.∇)
                param_w = layer.weights.output # The actual weights
                grad_w = layer.weights.∇       # The gradient for weights
                
                # apply! modifies the gradient array passed to it.
                # We must pass a copy if we don't want the original .∇ to be changed to the update step.
                grad_w_copy = copy(grad_w) 
                
                step_w = apply!(opt, param_w, grad_w_copy) # grad_w_copy is modified in-place by apply!
                param_w .-= step_w # Apply the update
            else
                # This might happen if weights are not part of the computational graph leading to the loss
                # or if backward pass did not populate the gradient.
                # @warn "Gradient for weights of layer $(hasfield(typeof(layer), :name) ? layer.name : "unnamed") is nothing or not defined. Skipping update."
            end

            # Update bias (if it exists and has a gradient)
            if !isnothing(layer.bias)
                if isdefined(layer.bias, :∇) && !isnothing(layer.bias.∇)
                    param_b = layer.bias.output # The actual biases
                    grad_b = layer.bias.∇       # The gradient for biases
                    
                    grad_b_copy = copy(grad_b)
                    
                    step_b = apply!(opt, param_b, grad_b_copy) # grad_b_copy is modified in-place
                    param_b .-= step_b # Apply the update
                else
                    # @warn "Gradient for bias of layer $(hasfield(typeof(layer), :name) ? layer.name : "unnamed") is nothing or not defined. Skipping update."
                end
            end
        # elseif isa(layer, SomeOtherTrainableLayer)
            # Handle parameters of other trainable layer types here
        end
    end
    return nothing
end