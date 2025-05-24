# https://github.com/FluxML/Flux.jl/blob/0e36af98f6fc5b7f3c95fe819a02172cfaaaf777/src/train.jl#L20
abstract type AbstractOptimiser end
const EPS = eps(Float32)

function optimize!(opt::AbstractOptimiser, params::Vector{Variable}, grads::Vector)
    for (i, param) in enumerate(params)
        apply!(opt, param, grads[i])
    end
end

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

# Corrected RMSProp optimizer (your current implementation has a bug)
mutable struct RMSProp <: AbstractOptimiser
    η::Float32    # learning rate
    ρ::Float32    # decay rate
    ϵ::Float32    # small constant for numerical stability
    state::IdDict{Any, Array{Float32}}   # stores moving average of squared gradients
    
    function RMSProp(η::Float32 = 0.001f0, ρ::Float32 = 0.9f0, ϵ::Float32 = EPS)
        new(η, ρ, ϵ, IdDict())
    end
end

function apply!(opt::RMSProp, param::Variable, grad::Array{Float32})
    # Initialize state if necessary (fixed: use param as key, not undefined variable)
    if !haskey(opt.state, param)
        opt.state[param] = zeros(Float32, size(param.output))
    end
    
    # Update moving average of squared gradients
    opt.state[param] .= opt.ρ .* opt.state[param] .+ (1.0f0 - opt.ρ) .* grad.^2
    
    # Update parameters
    param.output .-= opt.η .* grad ./ (sqrt.(opt.state[param] .+ opt.ϵ))
end