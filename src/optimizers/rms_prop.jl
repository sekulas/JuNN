using JuAD

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
    
    opt.state[param] .= opt.ρ .* opt.state[param] .+ (1.0f0 - opt.ρ) .* grad.^2
    
    param.output .-= opt.η .* grad ./ (sqrt.(opt.state[param]) .+ opt.ϵ)
end