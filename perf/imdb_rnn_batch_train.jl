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

# X_train = load("./data_rnn/imdb_dataset_prepared.jld2", "X_train")
# y_train = load("./data_rnn/imdb_dataset_prepared.jld2", "y_train")
# X_test = load("./data_rnn/imdb_dataset_prepared.jld2", "X_test")
# y_test = load("./data_rnn/imdb_dataset_prepared.jld2", "y_test")
# embeddings = load("./data_rnn/imdb_dataset_prepared.jld2", "embeddings")
# vocab = load("./data_rnn/imdb_dataset_prepared.jld2", "vocab")

#### DEBUGGING VERSION ####
# X_train = load("./data_rnn/imdb_dataset_prepared_bool_labels.jld2", "X_train")[:,1:10]
# y_train = load("./data_rnn/imdb_dataset_prepared_bool_labels.jld2", "y_train")[:,1:10]
# X_test = load("./data_rnn/imdb_dataset_prepared_bool_labels.jld2", "X_test")[:,1:10]
# y_test = load("./data_rnn/imdb_dataset_prepared_bool_labels.jld2", "y_test")[:,1:10]
# embeddings = load("./data_rnn/imdb_dataset_prepared.jld2", "embeddings")
# vocab = load("./data_rnn/imdb_dataset_prepared.jld2", "vocab")

sequence_length = 3
num_train = 3
num_test = 3
embed_dim = 5
vocab_size = 7

X_train = rand(1:vocab_size, sequence_length, num_train)
y_train = rand(Bool, 1, num_train)
X_test = rand(1:vocab_size, sequence_length, num_test)
y_test = rand(Bool, 1, num_test)
embeddings = randn(embed_dim, vocab_size)
vocab = [string("word", i) for i in 1:vocab_size]


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

net = NeuralNetwork(model, RMSProp(), binary_cross_entropy, accuracy; seq_length=sequence_length)

epochs = 12
for epoch in 1:epochs
    t = @elapsed begin
        train_loss, train_acc = train!(net, dataset)
    end
    
    test_loss, test_acc = evaluate(net, testset)
    @printf("Epoch %d/%d: Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f, Time: %.2fs\n",
            epoch, epochs, train_loss, train_acc, test_loss, test_acc, t)
end