module JuMLP
include("../src/structs.jl")
include("../src/scalar_operators.jl")
include("../src/broadcast_operators.jl")
include("../src/graph.jl")
include("../src/forward.jl")
include("../src/backward.jl")
include("../src/data_loader.jl")
include("../src/nn_model.jl")
# using JLD2
# X_train = load("./data/imdb_dataset_prepared.jld2", "X_train")
# y_train = load("./data/imdb_dataset_prepared.jld2", "y_train")
# X_test = load("./data/imdb_dataset_prepared.jld2", "X_test")
# y_test = load("./data/imdb_dataset_prepared.jld2", "y_test")
# X = (X_train, y_train)

function build_and_run_network(model::Network, x::Variable, y::Variable; epochs=10, lr=0.01f0)
    losses = Float64[]
    
    prediction = model(x)
    loss = mse_loss(prediction, y)
    loss.name = "loss"
    graph = topological_sort(loss)
    
    #println("Graph structure:")
    #for (i, node) in enumerate(graph)
    #    print(i, ". "); println(node)
    #end
    
    for epoch in 1:epochs  
        current_loss = forward!(graph)
        backward!(graph)
        
        update_params!(model, lr)
        
        loss_value = first(current_loss)
        push!(losses, loss_value)
        #println("Epoch $epoch, Loss: $loss_value")
    end    
    return prediction, losses
end

function run_example()
    model = Network(
        Dense(2 => 10, σ, name="dense1"),
        Dense(10 => 1, name="ŷ"),
    );

    x = Variable([1.98, 4.434], name="x")
    y = Variable([0.064], name="y")

    final_pred, losses = 
        build_and_run_network(model, x, y, epochs=1000, lr=0.005f0)
    println("\nFinal prediction: $(final_pred.output)")
    println("Target: $(y.output)")
    println("Final loss: $(last(losses))")
    
    return model, final_pred, losses
end

# Run example
model, prediction, losses = run_example()
end