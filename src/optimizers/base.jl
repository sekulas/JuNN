using JuAD

abstract type AbstractOptimiser end
const EPS = eps(Float32)

function optimize!(net::NeuralNetwork, batch_size::Int)

    for param in net.params
        grad = param.∇

        if size(param.output, 2) == 1 && size(grad, 2) != 1
            grad = sum(grad; dims=2)    # (out,1)
        end

        grad ./= Float32(batch_size)

        apply!(net.optimizer, param, grad)
    end

end