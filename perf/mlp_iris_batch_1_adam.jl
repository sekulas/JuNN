include("../src/structs.jl")
include("../src/scalar_operators.jl")
include("../src/broadcast_operators.jl")
include("../src/graph.jl")
include("../src/forward.jl")
include("../src/backward.jl")
include("../src/data_loader.jl")
include("../src/nn_model.jl")
include("../src/optimisers.jl")

using Base.Iterators: take, drop
using Statistics: mean
using Random: seed!

function calculate_batch_loss(batch, sorted_loss_graph, x_node::Variable, y_node::Variable)
    x_batched, y_batched = batch
    x_node.output .= x_batched
    y_node.output .= y_batched
    L = forward!(sorted_loss_graph)
    return L
end

function calculate_batch_accuracy(batch,
                                  y_model_prediction_node::GraphNode,
                                  sorted_graph_for_prediction,
                                  x_node::Variable)
    x_batched, y_batched = batch
    x_node.output .= x_batched   
    forward!(sorted_graph_for_prediction)
    y_pred_values = y_model_prediction_node.output
    pred_class_idx = argmax(y_pred_values)
    true_class_idx = argmax(y_batched)
    correct = (pred_class_idx == true_class_idx) ? 1 : 0
    return Float32(correct)
end


# Replace the old update_params! function
# The old function: update_params!(model::Network, lr::Float32; grads::Any) is removed.
# New function using Adam optimizer:
"""
    update_params!(model::Network, opt::Adam)

Update model parameters using the Adam optimizer.
This function iterates through the layers of the network, and for each trainable parameter
(weights and biases of Dense layers), it computes and applies the update using `apply!(opt, ...)`.
It assumes that gradients have been computed and stored in the `.∇` field of each parameter Variable
(e.g., `layer.weights.∇`, `layer.bias.∇`) by the `backward!` pass.
"""
function update_params!(model::Network, opt::Adam)
    for layer in model.layers
        if isa(layer, Dense)
            # Update weights
            if isdefined(layer.weights, :∇) && !isnothing(layer.weights.∇)
                param_w = layer.weights.output # The actual weights
                grad_w = layer.weights.∇       # The gradient for weights
                
                # apply! modifies the gradient array passed to it.
                # We must pass a copy if we don't want the original .∇ to be changed to the update step.
                grad_w_copy = copy(grad_w) 
                
                step_w = apply!(opt, param_w, grad_w_copy) # grad_w_copy is modified in-place by apply!
                param_w .-= step_w # Apply the update
            else
                # This might happen if weights are not part of the computational graph leading to the loss
                # or if backward pass did not populate the gradient.
                # @warn "Gradient for weights of layer $(hasfield(typeof(layer), :name) ? layer.name : "unnamed") is nothing or not defined. Skipping update."
            end

            # Update bias (if it exists and has a gradient)
            if !isnothing(layer.bias)
                if isdefined(layer.bias, :∇) && !isnothing(layer.bias.∇)
                    param_b = layer.bias.output # The actual biases
                    grad_b = layer.bias.∇       # The gradient for biases
                    
                    grad_b_copy = copy(grad_b)
                    
                    step_b = apply!(opt, param_b, grad_b_copy) # grad_b_copy is modified in-place
                    param_b .-= step_b # Apply the update
                else
                    # @warn "Gradient for bias of layer $(hasfield(typeof(layer), :name) ? layer.name : "unnamed") is nothing or not defined. Skipping update."
                end
            end
        # elseif isa(layer, SomeOtherTrainableLayer)
            # Handle parameters of other trainable layer types here
        end
    end
    return nothing
end

# Modify train_on_batch! to use the optimizer
function train_on_batch!(batch, model::Network, opt::Adam, sorted_loss_graph, x_node::Variable, y_node::Variable)
    x_batched, y_batched = batch
    x_node.output .= x_batched
    y_node.output .= y_batched
    
    forward!(sorted_loss_graph) # Compute loss and intermediate values
    backward!(sorted_loss_graph) # Compute gradients and store them in param.∇
    
    # The gradient(model) function collected param.∇ into a list.
    # The new update_params! directly uses param.∇, so explicit collection is not needed for the update step itself.
    # model_grads = gradient(model) # This line is no longer strictly needed here if update_params! uses .∇

    update_params!(model, opt) # Apply updates using Adam
    return nothing
end


# Modify learn! to instantiate and use Adam
function learn!(dataset_loader, # DataLoader object
                model::Network,
                sorted_loss_graph,             # For training and loss calculation
                y_model_prediction_node::GraphNode, # For accuracy calculation, m(x) node
                x_node::Variable,
                y_node::Variable;              # For loss calculation's true label placeholder
                optimizer::AbstractOptimiser,
                epochs::Int,
                print_after_iters::Int = 100)

    total_samples = length(dataset_loader)
    n_train = floor(Int, 0.7 * total_samples)
    # n_test = total_samples - n_train # User's original calculation
    # Ensure n_test is not negative if total_samples is small, and handle n_train=0 correctly
    if total_samples == 0
        n_train = 0
        n_test = 0
        @warn "Dataset is empty. Training and testing will be skipped."
    elseif n_train == 0 && total_samples > 0 # e.g. total_samples = 1
         @warn "n_train is 0 due to floor(0.7 * $total_samples). Consider a larger dataset or different split. Forcing 1 train sample if possible."
         n_train = min(1, total_samples) # Ensure at least one training sample if possible
         n_test = total_samples - n_train
    else
        n_test = total_samples - n_train
    end
    
    println("→ Dataset: $total_samples samples. Split: $n_train train, $n_test test.")

    t_start = time()
    total_iterations_ran = 0

    for epoch in 1:epochs
        println("\n=== Epoch $epoch/$epochs ===")

        epoch_train_total_loss = 0.0f0
        epoch_train_total_correct = 0
        num_train_batches_processed = 0
        periodic_train_losses = Float32[]
        periodic_train_accuracies = Float32[]
        t_epoch_train = time()

        # Resetting DataLoader is important if it's stateful and iterated multiple times
        # Assuming dataset_loader can be iterated freshly or reset for each epoch's split
        # If not, data might need to be pre-split or DataLoader handled differently.
        # For simplicity, let's assume it behaves like a resettable iterator or a collection.
        
        # Create iterators for the current epoch's train/test split
        # It's safer to re-create iterators each epoch if DataLoader's state after partial iteration is uncertain
        # If DataLoader shuffles, ensure it's re-shuffled or state is reset if desired per epoch
        
        # For train_batch_iterator, ensure it's only n_train items
        # If dataset_loader is an iterator that gets consumed:
        # Option 1: Collect all data and then index/slice (memory intensive for large datasets)
        # Option 2: Re-instantiate or reset DataLoader for each part (train/test)
        # Option 3: Carefully use take and drop, ensuring DataLoader supports this well for repeated calls
        
        # Assuming dataset_loader is iterable and supports `take` correctly without needing reset for this split logic
        current_epoch_loader_train = take(dataset_loader, n_train) # May need reset logic for dataset_loader for subsequent epochs
        
        if n_train > 0
            for batch in current_epoch_loader_train # Use the specific iterator for this epoch's training
                # Accuracy and Loss are calculated *before* the current batch's training step updates the model
                acc = calculate_batch_accuracy(batch, y_model_prediction_node, sorted_loss_graph, x_node)
                loss_val = calculate_batch_loss(batch, sorted_loss_graph, x_node, y_node)
                
                # Perform training step (forward, backward, optimizer update)
                # Note: train_on_batch! internally does forward and backward for the training step
                train_on_batch!(batch, model, optimizer, sorted_loss_graph, x_node, y_node)


                push!(periodic_train_losses, loss_val)
                push!(periodic_train_accuracies, acc)
                epoch_train_total_loss += loss_val
                epoch_train_total_correct += Int(acc) # acc is 0.0 or 1.0
                num_train_batches_processed += 1
                total_iterations_ran += 1

                if num_train_batches_processed % print_after_iters == 0 || num_train_batches_processed == n_train
                    avg_periodic_loss = isempty(periodic_train_losses) ? NaN32 : mean(periodic_train_losses)
                    avg_periodic_acc  = isempty(periodic_train_accuracies) ? NaN32 : mean(periodic_train_accuracies)
                    println("  [Train iter $num_train_batches_processed/$n_train]  Loss: $(round(avg_periodic_loss, digits=4)),  Acc: $(round(avg_periodic_acc, digits=4))")
                    empty!(periodic_train_losses)
                    empty!(periodic_train_accuracies)
                end
            end
        end
        train_time = time() - t_epoch_train

        epoch_test_total_loss = 0.0f0
        epoch_test_total_correct = 0
        num_test_batches_processed = 0
        t_epoch_test = time()

        # For test_batch_iterator
        # Using drop assumes dataset_loader can be "fast-forwarded" or iterated from a point.
        # If dataset_loader is a collection dataset_loader.data[n_train+1:end] might be an alternative.
        # The Iterators.rest(dataset_loader, n_train + 1) or drop(dataset_loader, n_train) logic
        # can be tricky if dataset_loader is a stateful iterator that was partially consumed by `take`.
        # A robust way is to have dataset_loader able to provide distinct iterators for train/test splits.
        # For now, let's assume `drop` works on a "fresh" or resettable `dataset_loader` conceptual view for this part.
        # If the DataLoader is consumed by `take`, `drop` on the same instance won't work as expected.
        # This part requires careful handling of how dataset_loader provides data.
        # A common pattern:
        # 1. Get all data indices.
        # 2. Shuffle indices.
        # 3. Split indices into train_indices and test_indices.
        # 4. For each epoch:
        #    Iterate using train_indices on the underlying data.
        #    Iterate using test_indices on the underlying data.
        # The current DataLoader might not support this directly. The `take` and `drop` approach from the user's code is maintained with a note of caution.

        # To make `drop` work after `take` on a stateful iterator, you'd typically need separate iterators or a resettable one.
        # If your DataLoader is just a wrapper around a tuple of arrays, (X_data, Y_data), then `take` and `drop`
        # operate on an iterator produced from it. For each epoch, if `dataset_loader` itself is passed, new iterators are made.
        current_epoch_loader_test = drop(dataset_loader, n_train)


        if n_test > 0
            actual_test_batches_iterated = 0
            for batch in current_epoch_loader_test # Use the specific iterator for this epoch's testing
                # Safety break if drop doesn't limit to n_test items and iterates more
                actual_test_batches_iterated += 1
                 if actual_test_batches_iterated > n_test # n_test could be 0, but loop condition n_test > 0 handles that
                     break
                 end

                acc = calculate_batch_accuracy(batch, y_model_prediction_node, sorted_loss_graph, x_node) # sorted_loss_graph is also used for prediction graph here
                loss_val = calculate_batch_loss(batch, sorted_loss_graph, x_node, y_node)
                epoch_test_total_loss += loss_val
                epoch_test_total_correct += Int(acc)
                num_test_batches_processed += 1
            end
        end
        test_time = time() - t_epoch_test

        avg_epoch_train_loss = num_train_batches_processed > 0 ? epoch_train_total_loss / num_train_batches_processed : NaN32
        avg_epoch_train_acc  = num_train_batches_processed > 0 ? Float32(epoch_train_total_correct / num_train_batches_processed) : NaN32
        avg_epoch_test_loss = num_test_batches_processed > 0 ? epoch_test_total_loss / num_test_batches_processed : NaN32
        avg_epoch_test_acc  = num_test_batches_processed > 0 ? Float32(epoch_test_total_correct / num_test_batches_processed) : NaN32

        println("  → Train (Epoch $epoch): $(round(train_time, digits=2))s | Loss: $(round(avg_epoch_train_loss, digits=4)), Acc: $(round(avg_epoch_train_acc, digits=4)) (Batches: $num_train_batches_processed)")
        println("  → Eval  (Epoch $epoch): $(round(test_time,  digits=2))s | Loss: $(round(avg_epoch_test_loss, digits=4)), Acc: $(round(avg_epoch_test_acc, digits=4)) (Batches: $num_test_batches_processed)")
    end

    total_time = time() - t_start
    println("\nTraining complete. Total time: ", round(total_time, digits=2), " seconds. Total iterations: $total_iterations_ran.")
end


seed!(0)

include("../data/iris.jl")

input_neurons = 4
hidden_neurons = 8
output_neurons = 3

X_data = Float32.(inputs')  # features x N_samples
Y_data = Float32.(targets') # classes x N_samples

println("Data loaded and processed for training.")
println("X_data shape: ", size(X_data))
println("Y_data shape: ", size(Y_data))

epochs_val = 900

x_node = Variable(zeros(Float32, input_neurons, 1), name="x_input_sample")
y_node = Variable(zeros(Float32, output_neurons, 1), name="y_true_label_sample")

dataset_loader = DataLoader((X_data, Y_data), batchsize=1, shuffle=true)
println("DataLoader created. Effective total samples (batches): ", length(dataset_loader))

model = Network(
    Dense((input_neurons => hidden_neurons), name="hidden_dense"),
    Dense((hidden_neurons => output_neurons), softmax, name="softmax_dense")
)

y_model_prediction_node = model(x_node)
loss_fn_output_node = cross_entropy_loss(y_node, y_model_prediction_node)
sorted_loss_graph = topological_sort(loss_fn_output_node)

println("Computation graphs sorted. Starting training...")
num_train_samples_approx = floor(Int, 0.7 * length(dataset_loader))
print_iters = num_train_samples_approx > 0 ? max(1, div(num_train_samples_approx, 5)) : 1

learn!(dataset_loader, model,
       sorted_loss_graph,
       y_model_prediction_node,
       x_node, y_node;
       optimizer=Adam(),
       epochs=epochs_val, print_after_iters=print_iters)