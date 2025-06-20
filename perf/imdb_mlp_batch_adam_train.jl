using JuNN

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

accuracy(y_true, y_pred) = mean((y_true .> 0.5f0) .== (y_pred .> 0.5f0))

net = NeuralNetwork(model, Adam(), binary_cross_entropy, accuracy, batch_size)

epochs = 5
for epoch in 1:epochs
    t = @elapsed begin
        train_loss, train_acc = train!(net, dataset)
    end
    
    test_loss, test_acc = evaluate(net, testset)
    @printf("Epoch %d/%d: Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f, Time: %.2fs\n",
            epoch, epochs, train_loss, train_acc, test_loss, test_acc, t)
end