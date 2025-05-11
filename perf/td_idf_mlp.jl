include("../src/structs.jl")
include("../src/scalar_operators.jl")
include("../src/broadcast_operators.jl")
include("../src/graph.jl")
include("../src/forward.jl")
include("../src/backward.jl")
include("../src/data_loader.jl")
include("../src/nn_model.jl")



using JLD2

###
# X_train = load("./data/imdb_dataset_prepared.jld2", "X_train")
# y_train = load("./data/imdb_dataset_prepared.jld2", "y_train")
# X_test = load("./data/imdb_dataset_prepared.jld2", "X_test")
# y_test = load("./data/imdb_dataset_prepared.jld2", "y_test")
###

# ####
using Random: seed!
seed!(42)
X_train = rand(Float32, 100, 80)  # (100, 10)
y_train = rand([0, 1], 1, 80)       # (100,)
X_test = rand(Float32, 100, 20)    # (10, 10)
y_test = rand([0, 1], 1, 20)         # (10,)
# ####

println("Data loaded successfully.")
println("X_train size: ", size(X_train))
println("y_train size: ", size(y_train))
println("X_test size: ", size(X_test))
println("y_test size: ", size(y_test))

loss(m, x, y) = binary_cross_entropy(m(x), y)
accuracy(m, x, y) = mean((m(x) .> 0.5) .== (y .> 0.5))

function test(batch, sorted_graph)
    x_batched, y_batched = batch

    x.output .= x_batched
    y.output .= y_batched

    L = forward!(sorted_graph)
    return L
end


function train!(batch, model, graph, lr=0.001f0)
    x_batched, y_batched = batch
    x.output .= x_batched
    y.output .= y_batched
    forward!(graph)
    backward!(graph)

    grads = gradient(model)

    # for j in Iterators.drop(batch, 2) #TODO!!!: CHECK AS BATCHED 2 ELEMS
    #     x.output .= inputs[j,:]
    #     y.output .= targets[j,:]

    #     forward!(graph)
    #     backward!(graph)
        
    #     accumulate_gradients!(grads, gradient(model))
    # end


    # batch_len::Integer = length(batch) / 2 # TODO!!!: ZMIEN TO

    update_params!(model, lr; grads, batch_len=1)
    # forward!(graph)
    return nothing
end

function accuracy_batch(batch, sorted_graph)
    x_batched, y_batched = batch

    x.output .= x_batched
    forward!(sorted_graph)

    y_pred = y.output

    return mean((y_pred .> 0.5) .== (y_batched .> 0.5))
end



function learn!(dataset, model, sorted_graph;
                lr, epochs::Int,
                print_after::Int = 1000)

    losses = Float32[]
    accs   = Float32[]
    current_count = 0
    t_start = time()

    println("Training started...")
    for epoch in 1:epochs
        t_epoch_start = time()

        # optionally decay lr
        if epoch % 4 == 0
            lr /= 10
            println("  → Epoch $epoch: decayed lr to $lr")
        end

        for batch in dataset
            # 1) do one gradient step
            train!(batch, model, sorted_graph, lr)

            # 2) measure loss & accuracy on that batch
            loss_val = test(batch, sorted_graph)
            acc_val  = accuracy_batch(batch, sorted_graph)

            push!(losses, loss_val)
            push!(accs,    acc_val)
            current_count += 1

            # 3) periodic print
            if current_count % print_after == 0
                println("  iter $current_count → loss = $(mean(losses)) | acc = $(mean(accs))")
                empty!(losses)
                empty!(accs)
            end
        end

        # epoch summary
        epoch_time = time() - t_epoch_start
        println("Epoch $epoch finished in $(round(epoch_time, digits=2))s; ")
        empty!(losses)
        empty!(accs)
        current_count = 0
    end

    total_time = time() - t_start
    println("All epochs done in $(round(total_time, digits=2)) seconds.")
end

epochs = 5
η = 0.01f0
dataset = DataLoader((X_train, y_train), batchsize=1, shuffle=true)
input_neurons = size(X_train, 1)
model = Network(
    Dense((input_neurons => 32), ReLU, name="ReLU_dense"),
    Dense((32 => 1), σ, name="σ_dense")
)

x = Variable(zeros(Float32, input_neurons, 1), name="x")
y = Variable(zeros(Float32, 1, 1), name="y")
graph = loss(model, x, y)
sorted_graph = topological_sort(graph)



learn!(dataset, model, sorted_graph; lr=η, epochs, print_after=1000)
