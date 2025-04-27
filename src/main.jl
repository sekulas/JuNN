module JuMLP
include("structs.jl")
include("scalar_operators.jl")
include("graph.jl")
include("forward.jl")
include("backward.jl")

# using JLD2
# X_train = load("./data/imdb_dataset_prepared.jld2", "X_train")
# y_train = load("./data/imdb_dataset_prepared.jld2", "y_train")
# X_test = load("./data/imdb_dataset_prepared.jld2", "X_test")
# y_test = load("./data/imdb_dataset_prepared.jld2", "y_test")
# X = (X_train, y_train)


end
