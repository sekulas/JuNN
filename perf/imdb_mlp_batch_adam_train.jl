include("../src/structs.jl")
include("../src/scalar_operators.jl")
include("../src/broadcast_operators.jl")
include("../src/graph.jl")
include("../src/forward.jl")
include("../src/backward.jl")
include("../src/data_loader.jl")
include("../src/nn_model.jl")
include("../src/optimisers.jl")
include("../src/losses.jl")
include("../src/layers.jl")
include("../src/recurrent.jl")

using Printf, Random
Random.seed!(0)

using JLD2

#### DEBUGGING VERSION ####
#X_train = load("./data/imdb_dataset_prepared_bool_labels.jld2", "X_train")
#y_train = load("./data/imdb_dataset_prepared_bool_labels.jld2", "y_train")
#X_test = load("./data/imdb_dataset_prepared_bool_labels.jld2", "X_test")
#y_test = load("./data/imdb_dataset_prepared_bool_labels.jld2", "y_test")

X_train = load("./data/imdb_dataset_prepared.jld2", "X_train")
y_train = load("./data/imdb_dataset_prepared.jld2", "y_train")
X_test = load("./data/imdb_dataset_prepared.jld2", "X_test")
y_test = load("./data/imdb_dataset_prepared.jld2", "y_test")
println("X_train: ", size(X_train))
println("y_train: ", size(y_train))
println("X_test: ", size(X_test))
println("y_test: ", size(y_test))

X_train = Float32.(X_train)
y_train = Float32.(y_train)
X_test = Float32.(X_test)
y_test = Float32.(y_test)

batch_size = 64

dataset = DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)
testset = DataLoader((X_test, y_test), batchsize=batch_size, shuffle=false)

input_size = size(X_train, 1)

model = Chain(
    Dense((input_size => 32), ReLU, name="hidden_layer"),
    Dense((32 => 1), σ, name="output_layer")
)

accuracy(y_true, y_pred) = mean((y_true .> 0.5) .== (y_pred .> 0.5))

net = NeuralNetwork(model, Adam(), binary_cross_entropy, accuracy)

epochs = 5
for epoch in 1:epochs
    t = @elapsed begin
        train_loss, train_acc = @time train!(net, dataset)
    end
    
    test_loss, test_acc = evaluate(net, testset)
    @printf("Epoch %d/%d: Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f, Time: %.2fs\n",
            epoch, epochs, train_loss, train_acc, test_loss, test_acc, t)
end

# X_train: (17703, 8000)
# y_train: (1, 8000)
# X_test: (17703, 2000)
# y_test: (1, 2000)
# Epoch 1/5: Train Loss: 0.6406, Train Acc: 0.7986, Test Loss: 0.5710, Test Acc: 0.8389, Time: 31.46s
# Epoch 2/5: Train Loss: 0.4496, Train Acc: 0.9146, Test Loss: 0.4339, Test Acc: 0.8608, Time: 20.95s
# Epoch 3/5: Train Loss: 0.2909, Train Acc: 0.9433, Test Loss: 0.3627, Test Acc: 0.8647, Time: 24.42s
# Epoch 4/5: Train Loss: 0.1984, Train Acc: 0.9643, Test Loss: 0.3311, Test Acc: 0.8721, Time: 24.70s
# Epoch 5/5: Train Loss: 0.1416, Train Acc: 0.9774, Test Loss: 0.3186, Test Acc: 0.8726, Time: 22.26s


# UPDATE! UPDATED ADDED BY CHAT + BATCH ADDED BY CHAT
#  14.167882 seconds (15.15 M allocations: 3.143 GiB, 3.92% gc time, 81.48% compilation time)
# Epoch 1/5: Train Loss: 0.6449, Train Acc: 0.7997, Test Loss: 0.5798, Test Acc: 0.8354, Time: 14.20s
#   2.337058 seconds (101.75 k allocations: 2.387 GiB, 3.72% gc time)
# Epoch 2/5: Train Loss: 0.4650, Train Acc: 0.9095, Test Loss: 0.4435, Test Acc: 0.8623, Time: 2.34s
#   2.208099 seconds (101.75 k allocations: 2.387 GiB, 2.68% gc time)
# Epoch 3/5: Train Loss: 0.3068, Train Acc: 0.9396, Test Loss: 0.3683, Test Acc: 0.8647, Time: 2.21s
#   2.315478 seconds (101.75 k allocations: 2.387 GiB, 3.14% gc time)
# Epoch 4/5: Train Loss: 0.2121, Train Acc: 0.9609, Test Loss: 0.3335, Test Acc: 0.8716, Time: 2.32s
#   2.396177 seconds (101.75 k allocations: 2.387 GiB, 3.21% gc time)
# Epoch 5/5: Train Loss: 0.1533, Train Acc: 0.9745, Test Loss: 0.3187, Test Acc: 0.8735, Time: 2.40s

# NO CHAT OPTIMIZATIONS - PREVIOUS VERSION
#  41.051407 seconds (23.29 M allocations: 19.266 GiB, 3.91% gc time, 38.44% compilation time)
# Epoch 1/5: Train Loss: 0.6412, Train Acc: 0.7985, Test Loss: 0.5724, Test Acc: 0.8389, Time: 41.10s
#  25.405343 seconds (5.28 M allocations: 18.368 GiB, 4.30% gc time)
# Epoch 2/5: Train Loss: 0.4518, Train Acc: 0.9143, Test Loss: 0.4355, Test Acc: 0.8599, Time: 25.41s
#  24.267112 seconds (5.28 M allocations: 18.368 GiB, 4.17% gc time)
# Epoch 3/5: Train Loss: 0.2932, Train Acc: 0.9431, Test Loss: 0.3638, Test Acc: 0.8647, Time: 24.27s
#  23.837214 seconds (5.28 M allocations: 18.368 GiB, 4.34% gc time)
# Epoch 4/5: Train Loss: 0.2004, Train Acc: 0.9634, Test Loss: 0.3317, Test Acc: 0.8726, Time: 23.84s
#  24.977042 seconds (5.28 M allocations: 18.368 GiB, 4.47% gc time)
# Epoch 5/5: Train Loss: 0.1433, Train Acc: 0.9770, Test Loss: 0.3188, Test Acc: 0.8730, Time: 24.98s

# UPDATE! UPDATED ADDED BY CHAT
#  42.564746 seconds (23.62 M allocations: 36.693 GiB, 5.07% gc time, 30.48% compilation time)
# Epoch 1/5: Train Loss: 0.6412, Train Acc: 0.7985, Test Loss: 0.5724, Test Acc: 0.8389, Time: 42.61s
#  28.625426 seconds (5.65 M allocations: 35.797 GiB, 5.72% gc time)
# Epoch 2/5: Train Loss: 0.4518, Train Acc: 0.9143, Test Loss: 0.4355, Test Acc: 0.8599, Time: 28.63s
#  28.987382 seconds (5.65 M allocations: 35.797 GiB, 6.27% gc time)
# Epoch 3/5: Train Loss: 0.2932, Train Acc: 0.9431, Test Loss: 0.3638, Test Acc: 0.8647, Time: 28.99s
#  28.687725 seconds (5.65 M allocations: 35.797 GiB, 5.59% gc time)
# Epoch 4/5: Train Loss: 0.2004, Train Acc: 0.9634, Test Loss: 0.3317, Test Acc: 0.8726, Time: 28.69s
#  28.987223 seconds (5.65 M allocations: 35.797 GiB, 5.47% gc time)
# Epoch 5/5: Train Loss: 0.1433, Train Acc: 0.9770, Test Loss: 0.3188, Test Acc: 0.8730, Time: 28.99s

# BATCH ADDED BY CHAT
#  15.196715 seconds (15.14 M allocations: 2.347 GiB, 3.74% gc time, 80.90% compilation time)
# Epoch 1/5: Train Loss: 0.6449, Train Acc: 0.7997, Test Loss: 0.5798, Test Acc: 0.8354, Time: 15.24s
#   2.524914 seconds (95.50 k allocations: 1.591 GiB, 3.29% gc time)
# Epoch 2/5: Train Loss: 0.4650, Train Acc: 0.9095, Test Loss: 0.4435, Test Acc: 0.8623, Time: 2.53s
#   3.066202 seconds (95.50 k allocations: 1.591 GiB, 2.78% gc time)
# Epoch 3/5: Train Loss: 0.3068, Train Acc: 0.9396, Test Loss: 0.3683, Test Acc: 0.8647, Time: 3.07s
#   2.411099 seconds (95.50 k allocations: 1.591 GiB, 2.96% gc time)
# Epoch 4/5: Train Loss: 0.2121, Train Acc: 0.9609, Test Loss: 0.3335, Test Acc: 0.8716, Time: 2.41s
#   2.126266 seconds (95.50 k allocations: 1.591 GiB, 2.51% gc time)
# Epoch 5/5: Train Loss: 0.1533, Train Acc: 0.9745, Test Loss: 0.3187, Test Acc: 0.8735, Time: 2.13s


# BATCH WITH FILL
#  16.861452 seconds (15.13 M allocations: 2.082 GiB, 3.55% gc time, 81.60% compilation time)
# Epoch 1/5: Train Loss: 0.6449, Train Acc: 0.7997, Test Loss: 0.5798, Test Acc: 0.8354, Time: 16.90s
#   2.348888 seconds (93.88 k allocations: 1.328 GiB, 2.64% gc time)
# Epoch 2/5: Train Loss: 0.4650, Train Acc: 0.9095, Test Loss: 0.4435, Test Acc: 0.8623, Time: 2.35s
#   2.816352 seconds (93.88 k allocations: 1.328 GiB, 1.50% gc time)
# Epoch 3/5: Train Loss: 0.3068, Train Acc: 0.9396, Test Loss: 0.3683, Test Acc: 0.8647, Time: 2.82s
#   2.309097 seconds (93.88 k allocations: 1.328 GiB, 1.61% gc time)
# Epoch 4/5: Train Loss: 0.2121, Train Acc: 0.9609, Test Loss: 0.3335, Test Acc: 0.8716, Time: 2.31s
#   2.359030 seconds (93.88 k allocations: 1.328 GiB, 1.99% gc time)
# Epoch 5/5: Train Loss: 0.1533, Train Acc: 0.9745, Test Loss: 0.3187, Test Acc: 0.8735, Time: 2.36s


# BATCH WITHOUT FILL
#  14.254571 seconds (15.13 M allocations: 2.082 GiB, 3.52% gc time, 84.56% compilation time)
# Epoch 1/5: Train Loss: 0.6449, Train Acc: 0.7997, Test Loss: 0.5798, Test Acc: 0.8354, Time: 14.29s
#   2.173898 seconds (93.88 k allocations: 1.328 GiB, 3.08% gc time)
# Epoch 2/5: Train Loss: 0.4650, Train Acc: 0.9095, Test Loss: 0.4435, Test Acc: 0.8623, Time: 2.17s
#   2.140900 seconds (93.88 k allocations: 1.328 GiB, 2.22% gc time)
# Epoch 3/5: Train Loss: 0.3068, Train Acc: 0.9396, Test Loss: 0.3683, Test Acc: 0.8647, Time: 2.14s
#   2.259611 seconds (93.88 k allocations: 1.328 GiB, 2.59% gc time)
# Epoch 4/5: Train Loss: 0.2121, Train Acc: 0.9609, Test Loss: 0.3335, Test Acc: 0.8716, Time: 2.26s
#   2.177831 seconds (93.88 k allocations: 1.328 GiB, 2.74% gc time)
# Epoch 5/5: Train Loss: 0.1533, Train Acc: 0.9745, Test Loss: 0.3187, Test Acc: 0.8735, Time: 2.18s
