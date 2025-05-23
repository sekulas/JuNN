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

X_train = load("./data_rnn/imdb_dataset_prepared.jld2", "X_train")
y_train = load("./data_rnn/imdb_dataset_prepared.jld2", "y_train")
X_test = load("./data_rnn/imdb_dataset_prepared.jld2", "X_test")
y_test = load("./data_rnn/imdb_dataset_prepared.jld2", "y_test")
embeddings = load("./data_rnn/imdb_dataset_prepared.jld2", "embeddings")
vocab = load("./data_rnn/imdb_dataset_prepared.jld2", "vocab")

#### DEBUGGING VERSION ####
# X_train = load("./data_rnn/imdb_dataset_prepared_bool_labels.jld2", "X_train")
# y_train = load("./data_rnn/imdb_dataset_prepared_bool_labels.jld2", "y_train")
# X_test = load("./data_rnn/imdb_dataset_prepared_bool_labels.jld2", "X_test")
# y_test = load("./data_rnn/imdb_dataset_prepared_bool_labels.jld2", "y_test")
# embeddings = load("./data_rnn/imdb_dataset_prepared.jld2", "embeddings")
# vocab = load("./data_rnn/imdb_dataset_prepared.jld2", "vocab")

println("X_train: ", size(X_train))
println("y_train: ", size(y_train))
println("X_test: ", size(X_test))
println("y_test: ", size(y_test))
println("embeddings: ", size(embeddings))
println("vocab: ", size(vocab))


embedding_dim = size(embeddings, 1)  # 50
vocab_size = length(vocab)           # 12849
seq_length = size(X_train, 1)        # 130


batch_size = 64

dataset = DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)

accuracy(y_true, y_pred) = mean((y_true .> 0.5) .== (y_pred .> 0.5))

model = Chain(
    Embedding(vocab_size, embedding_dim),           # Embedding layer
    RNN((embedding_dim => 16), ReLU; return_state=true), # RNN layer with 16 hidden units, ReLU activation
    get_last,                                    # Extract the last element (equivalent to x -> x[end])
    flatten,                                        # Flatten output
    Dense((16 => 1), σ)                               # Output layer with sigmoid activation
)

model.layers[1].weights.output .= embeddings;

net = NeuralNetwork(model, RMSProp(), binary_cross_entropy, accuracy)

epochs = 5
for epoch in 1:epochs
    t = @elapsed begin
        train_loss, train_acc = train!(net, dataset)
    end
    
    test_loss, test_acc = evaluate(net, testset)
    @printf("Epoch %d/%d: Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f, Time: %.2fs\n",
            epoch, epochs, train_loss, train_acc, test_loss, test_acc, t)
end