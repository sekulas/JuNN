include("../src/structs.jl")
include("../src/scalar_operators.jl")
include("../src/broadcast_operators.jl")
include("../src/graph.jl")
include("../src/forward.jl")
include("../src/backward.jl")
include("../src/data_loader.jl")
include("../src/nn_model.jl")
include("../src/optimisers.jl")

using Printf, Statistics, Random
using JLD2

# Load data
X_train = load("./data/imdb_dataset_prepared.jld2", "X_train")
y_train = load("./data/imdb_dataset_prepared.jld2", "y_train")
X_test = load("./data/imdb_dataset_prepared.jld2", "X_test")
y_test = load("./data/imdb_dataset_prepared.jld2", "y_test")

println("X_train: ", size(X_train))
println("y_train: ", size(y_train))
println("X_test: ", size(X_test))
println("y_test: ", size(y_test))

# Convert data to Float32 for consistency
X_train = Float32.(X_train)
y_train = Float32.(y_train)
X_test = Float32.(X_test)
y_test = Float32.(y_test)

# Set random seed
Random.seed!(42)

# Create data loader with batch size 1 (no batching)
batch_size = 1
dataset = DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)

# Create model
input_size = size(X_train, 1)
model = Network(
    Dense((input_size => 32), ReLU, name="hidden_layer"),
    Dense((32 => 1), σ, name="output_layer")
)

# Setup input/output nodes for computation graph
x_node = Variable(zeros(Float32, input_size, 1), name="x_input")
y_node = Variable(zeros(Float32, 1, 1), name="y_true")

# Build computation graph
y_pred_node = model(x_node)
loss_node = binary_cross_entropy(y_node, y_pred_node)
sorted_graph = topological_sort(loss_node)

# Define accuracy function for single samples
function accuracy_single(pred, actual)
    return (pred > 0.5f0) == (actual > 0.5f0) ? 1.0f0 : 0.0f0
end

# Create optimizer
optimizer = Adam()

epochs = 5
for epoch in 1:epochs
    total_loss = 0.0f0
    total_acc = 0.0f0
    num_samples = 0
    
    t = @elapsed begin
        for (x_sample, y_sample) in dataset
            x_node.output .= x_sample
            y_node.output .= y_sample
            
            forward!(sorted_graph)
            batch_loss = loss_node.output
            total_loss += batch_loss
            
            pred_value = y_pred_node.output[1]  
            actual_value = y_sample[1]          
            batch_acc = accuracy_single(pred_value, actual_value)
            total_acc += batch_acc
            
            backward!(sorted_graph)
            update_params!(model, optimizer)
            
            num_samples += 1
            if num_samples % 100 == 0
                println("Epoch: $epoch, Sample: $num_samples, Loss: $batch_loss, Acc: $batch_acc")
            end
        end
    end
    
    # Calculate average metrics
    train_loss = total_loss / num_samples
    train_acc = total_acc / num_samples
    
    # Evaluate on test set
    test_loss = 0.0f0
    test_correct = 0
    num_test_samples = size(X_test, 2)
    
    # Create test nodes with single sample size
    x_test_node = Variable(zeros(Float32, input_size, 1), name="x_test")
    y_test_node = Variable(zeros(Float32, 1, 1), name="y_test")
    
    # Create test computation graph
    test_pred_node = model(x_test_node)
    test_loss_node = binary_cross_entropy(y_test_node, test_pred_node)
    test_graph = topological_sort(test_loss_node)
    
    for i in 1:num_test_samples
        # Extract single test sample
        x_test_sample = X_test[:, i:i]
        y_test_sample = y_test[:, i:i]
        
        # Set test data
        x_test_node.output .= x_test_sample
        y_test_node.output .= y_test_sample
        
        # Forward pass for test sample
        forward!(test_graph)
        test_loss += test_loss_node.output
        
        # Calculate accuracy
        pred_value = test_pred_node.output[1]
        actual_value = y_test_sample[1]
        if (pred_value > 0.5f0) == (actual_value > 0.5f0)
            test_correct += 1
        end
    end
    
    # Calculate test metrics
    test_loss = test_loss / num_test_samples
    test_acc = Float32(test_correct) / num_test_samples
    
    # Print results
    println(@sprintf("Epoch: %d (%.2fs) \tTrain: (l: %.4f, a: %.4f) \tTest: (l: %.4f, a: %.4f)",
        epoch, t, train_loss, train_acc, test_loss, test_acc))
end
