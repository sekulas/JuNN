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

x = Variable(5.0, name="x")
two = Constant(2.0)
squared = x^two
sine = sin(squared)

order = topological_sort(sine)

y = forward!(order)

backward!(order)

end
