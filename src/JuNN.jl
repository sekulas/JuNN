module JuNN

using JuAD
using Random: GLOBAL_RNG, AbstractRNG

# Export layer types
export Dense, Chain, RNN, RNNCell, Embedding

# Export model types
export NeuralNetwork

# Export loss functions
export mse_loss, cross_entropy_loss, binary_cross_entropy

# Export optimizers
export Adam, RMSProp, apply!, optimize!

# Export data utilities
export DataLoader

# Export utility functions
export get_params, glorot_uniform, train!, evaluate

export
    ReLU,
    sigmoid,
    softmax,
    tanh,
    σ,
    mean


include("./layers/activations.jl")
include("./layers/dense.jl")
include("./layers/recurrent.jl")
include("./layers/embedding.jl")

include("./models/chain.jl")
include("./models/neural_network.jl")

include("./optimizers/base.jl")
include("./optimizers/adam.jl")
include("./optimizers/rms_prop.jl")

include("./utils/weights_gen.jl")

include("./losses.jl")
include("./data_loader.jl")

end