using JuAD
mutable struct Adam <: AbstractOptimiser
    eta::Float32
    beta::Tuple{Float32,Float32}
    epsilon::Float32
    state::IdDict{Any, Tuple{Any,Any,Vector{Float32},Any}}  # mt, vt, βp, Δ_stepend # TODO: TYPES
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
Note: This function does modifies param gradient.
"""
function apply!(o::Adam, param::Variable, Δ::AbstractArray)
    η, β = o.eta, o.beta

    mt, vt, βp, Δ_step = get!(o.state, param) do
        (zeros(size(param.output)), 
         zeros(size(param.output)), 
         Float32[β[1], β[2]],
         similar(param.output))
    end

    # Update biased first moment estimate
    @. mt = β[1] * mt + (1 - β[1]) * Δ
  
    # Update biased second raw moment estimate
    @. vt = β[2] * vt + (1 - β[2]) * Δ * Δ


    @. Δ_step =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon) * η
    βp .= βp .* β

    param.output .-= Δ_step
end