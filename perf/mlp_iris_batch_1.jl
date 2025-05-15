include("../src/structs.jl")
include("../src/scalar_operators.jl")
include("../src/broadcast_operators.jl")
include("../src/graph.jl")
include("../src/forward.jl")
include("../src/backward.jl")
include("../src/data_loader.jl")
include("../src/nn_model.jl")
using Base.Iterators: take, drop
using Statistics: mean
using Random: seed!
seed!(0)

function calculate_batch_loss(batch, sorted_loss_graph, x_node::Variable, y_node::Variable)
    x_batched, y_batched = batch
    x_node.output .= x_batched
    y_node.output .= y_batched
    L = forward!(sorted_loss_graph)
    return L
end

function train_on_batch!(batch, model, sorted_loss_graph, lr, x_node::Variable, y_node::Variable)
    x_batched, y_batched = batch
    x_node.output .= x_batched
    y_node.output .= y_batched
    forward!(sorted_loss_graph)
    backward!(sorted_loss_graph)
    model_grads = gradient(model)
    update_params!(model, lr; grads=model_grads, batch_len=1) # TODO: batch other than 1?
    return nothing
end

function calculate_batch_accuracy(batch,
                                  y_model_prediction_node::GraphNode, # The m(x) node
                                  sorted_graph_for_prediction, # Graph sorted up to y_model_prediction_node
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

function learn!(dataset_loader,
                model,
                sorted_loss_graph,             
                y_model_prediction_node,       
                x_node::Variable,
                y_node::Variable;              
                lr, epochs::Int,
                print_after_iters::Int = 100)

    total_samples = length(dataset_loader)
    n_train = floor(Int, 0.7 * total_samples)
    n_test = total_samples - n_train

    n_test = max(0, n_test)
    if n_train == 0 && total_samples > 0
        @warn "n_train is 0. Consider a larger dataset or different split."
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

        train_batch_iterator = take(dataset_loader, n_train)

        if n_train > 0
            for batch in train_batch_iterator
                acc = calculate_batch_accuracy(batch, y_model_prediction_node, sorted_loss_graph, x_node)
                loss_val = calculate_batch_loss(batch, sorted_loss_graph, x_node, y_node)
                train_on_batch!(batch, model, sorted_loss_graph, lr, x_node, y_node)

                push!(periodic_train_losses, loss_val)
                push!(periodic_train_accuracies, acc)
                epoch_train_total_loss += loss_val
                epoch_train_total_correct += Int(acc)
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

        test_batch_iterator = drop(dataset_loader, n_train)

        if n_test > 0
            actual_test_batches_iterated = 0
            for batch in test_batch_iterator

                actual_test_batches_iterated += 1
                if actual_test_batches_iterated > n_test && n_test > 0 # n_test could be 0
                     break
                end

                acc = calculate_batch_accuracy(batch, y_model_prediction_node, sorted_loss_graph, x_node)
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


include("../data/iris.jl")

input_neurons = 4
hidden_neurons = 8
output_neurons = 3

X_data = Float32.(inputs')  # features x N_samples
Y_data = Float32.(targets') # classes x N_samples

println("Data loaded and processed for training.")
println("X_data shape: ", size(X_data)) # Should be (4, N_samples)
println("Y_data shape: ", size(Y_data)) # Should be (3, N_samples)

epochs = 900
η = 0.01f0

x_node = Variable(zeros(Float32, input_neurons, 1), name="x_input_sample")
y_node = Variable(zeros(Float32, output_neurons, 1), name="y_true_label_sample")

dataset_loader = DataLoader((X_data, Y_data), batchsize=1, shuffle=true)
println("DataLoader created. Effective total samples (batches): ", length(dataset_loader))

model = Chain(
    Dense((input_neurons => hidden_neurons), name="hidden_dense"),
    Dense((hidden_neurons => output_neurons), softmax, name="softmax_dense")
)

y_model_prediction_node = model(x_node)
loss_fn_output_node = cross_entropy_loss(y_node, y_model_prediction_node)
sorted_loss_graph = topological_sort(loss_fn_output_node)

println("Computation graphs sorted. Starting training...")
num_train_samples_approx = floor(Int, 0.7 * length(dataset_loader))
print_iters = num_train_samples_approx > 0 ? max(1, div(num_train_samples_approx, 5)) : 1 # Print ~5 times during train part of epoch

learn!(dataset_loader, model,
       sorted_loss_graph,
       y_model_prediction_node,
       x_node, y_node;
       lr=η, epochs=epochs, print_after_iters=print_iters)