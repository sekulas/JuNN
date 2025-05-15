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

using Printf, Random
Random.seed!(0)

include("../data/iris.jl")

input_neurons = 4
hidden_neurons = 8
output_neurons = 3
batchsize = 4

X_data = Float32.(inputs')  # features x N_samples
Y_data = Float32.(targets') # classes x N_samples

# Split into train/test sets (e.g., 80% train, 20% test)
n_samples = size(X_data, 2)
n_train = Int(floor(0.7 * n_samples))
shuffle_idx = shuffle(1:n_samples)

train_idx = shuffle_idx[1:n_train]
test_idx = shuffle_idx[n_train+1:end]

X_train = X_data[:, train_idx]
Y_train = Y_data[:, train_idx]
X_test = X_data[:, test_idx]
Y_test = Y_data[:, test_idx]

println("Data loaded and processed for training.")
println("X_data shape: ", size(X_data))
println("Y_data shape: ", size(Y_data))

dataset = DataLoader((X_train, Y_train), batchsize=batchsize, shuffle=true)
testset = DataLoader((X_test, Y_test), batchsize=batchsize, shuffle=false)
println("DataLoader created. Effective total samples (batches): ", length(dataset))

model = Chain(
    Dense((input_neurons => hidden_neurons), name="hidden_dense"),
    Dense((hidden_neurons => output_neurons), softmax, name="softmax_dense")
)

function compute_accuracy(y_true, y_predicted)
    true_class_idx = argmax(y_true)
    pred_class_idx = argmax(y_predicted)
    correct = (pred_class_idx == true_class_idx) ? 1 : 0
    return Float32(correct)
end

net = NeuralNetwork(model, Adam(), cross_entropy_loss, compute_accuracy)

###########################################

epochs = 1000
for epoch in 1:epochs
    t = @elapsed begin
        train_loss, train_acc = train!(net, dataset)
    end
    
    test_loss, test_acc = evaluate(net, testset)
    @printf("Epoch %d/%d: Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f, Time: %.2fs\n",
            epoch, epochs, train_loss, train_acc, test_loss, test_acc, t)
end


