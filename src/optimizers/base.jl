using JuAD

abstract type AbstractOptimiser end
const EPS = eps(Float32)

function optimize!(opt::AbstractOptimiser, params::Vector{Variable}, grads::Vector)
    for (i, param) in enumerate(params)
        apply!(opt, param, grads[i])
    end
end