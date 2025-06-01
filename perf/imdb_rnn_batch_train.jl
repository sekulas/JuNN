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
# X_train = load("./data_rnn/imdb_dataset_prepared_bool_labels.jld2", "X_train")[:,1:10]
# y_train = load("./data_rnn/imdb_dataset_prepared_bool_labels.jld2", "y_train")[:,1:10]
# X_test = load("./data_rnn/imdb_dataset_prepared_bool_labels.jld2", "X_test")[:,1:10]
# y_test = load("./data_rnn/imdb_dataset_prepared_bool_labels.jld2", "y_test")[:,1:10]
# embeddings = load("./data_rnn/imdb_dataset_prepared.jld2", "embeddings")
# vocab = load("./data_rnn/imdb_dataset_prepared.jld2", "vocab")

# sequence_length = 3
# num_train = 3
# num_test = 3
# embed_dim = 5
# vocab_size = 7

# X_train = rand(1:vocab_size, sequence_length, num_train)
# y_train = rand(Bool, 1, num_train)
# X_test = rand(1:vocab_size, sequence_length, num_test)
# y_test = rand(Bool, 1, num_test)
# embeddings = randn(embed_dim, vocab_size)
# vocab = [string("word", i) for i in 1:vocab_size]


println("X_train: ", size(X_train))
println("y_train: ", size(y_train))
println("X_test: ", size(X_test))
println("y_test: ", size(y_test))
println("embeddings: ", size(embeddings))
println("vocab: ", size(vocab))

vocab_size = length(vocab)
embed_dim = size(embeddings, 1)  # 50 from your comment
sequence_length = size(X_train, 1)  # 130 from your comment
batch_size = 128

model = Chain(
    Embedding(vocab_size, embed_dim, name="embedding"),
    RNN(embed_dim, 16, return_sequences=false, name="rnn_layer"),
    Dense((16 => 1), σ, name="output_layer")
)
    
model.layers[1].weights.output .= embeddings
dataset = DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)
testset = DataLoader((X_test, y_test), batchsize=batch_size, shuffle=false)

accuracy(y_true, y_pred) = mean((y_true .> 0.5) .== (y_pred .> 0.5))

net = NeuralNetwork(model, RMSProp(), binary_cross_entropy, accuracy, batch_size, seq_length=sequence_length)

epochs = 12
for epoch in 1:epochs
    t = @elapsed begin
        train_loss, train_acc = train!(net, dataset)
    end
    
    test_loss, test_acc = evaluate(net, testset)
    @printf("Epoch %d/%d: Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f, Time: %.2fs\n",
            epoch, epochs, train_loss, train_acc, test_loss, test_acc, t)
    # @printf("Epoch %d/%d: Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f\n",
        # epoch, epochs, train_loss, train_acc, test_loss, test_acc)
end


# NO CLIPPING , NORMALIZATION
# Epoch 1/12: Train Loss: 0.6980, Train Acc: 0.5092, Test Loss: 0.6948, Test Acc: 0.5029, Time: 144.17s
# Epoch 2/12: Train Loss: 0.6907, Train Acc: 0.5209, Test Loss: 0.6934, Test Acc: 0.5067, Time: 126.95s
# Epoch 3/12: Train Loss: 0.6860, Train Acc: 0.5296, Test Loss: 0.6928, Test Acc: 0.5100, Time: 145.83s
# Epoch 4/12: Train Loss: 0.6759, Train Acc: 0.5623, Test Loss: 0.6710, Test Acc: 0.6227, Time: 150.83s
# Epoch 5/12: Train Loss: 0.6130, Train Acc: 0.6854, Test Loss: 0.6110, Test Acc: 0.6950, Time: 126.03s
# Epoch 6/12: Train Loss: 0.5532, Train Acc: 0.7391, Test Loss: 0.5409, Test Acc: 0.7589, Time: 125.34s
# Epoch 7/12: Train Loss: 0.5021, Train Acc: 0.7778, Test Loss: 0.5466, Test Acc: 0.7844, Time: 128.21s
# Epoch 8/12: Train Loss: 0.4644, Train Acc: 0.8011, Test Loss: 0.6189, Test Acc: 0.7998, Time: 126.52s
# Epoch 9/12: Train Loss: 0.4312, Train Acc: 0.8213, Test Loss: 0.4841, Test Acc: 0.8072, Time: 128.63s
# Epoch 10/12: Train Loss: 0.4090, Train Acc: 0.8340, Test Loss: 0.4525, Test Acc: 0.8128, Time: 123.02s
# Epoch 11/12: Train Loss: 0.3877, Train Acc: 0.8452, Test Loss: 0.4388, Test Acc: 0.8219, Time: 123.40s
# Epoch 12/12: Train Loss: 0.3705, Train Acc: 0.8542, Test Loss: 0.4220, Test Acc: 0.8269, Time: 123.83s