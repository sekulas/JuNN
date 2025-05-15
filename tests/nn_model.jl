using Test
include("../src/nn_model.jl")
include("../src/broadcast_operators.jl")
include("../src/graph.jl")
include("../src/forward.jl")
include("../src/backward.jl")
# TODO: Czy bias macierz czy wektor?
# @testset "create_bias tests" begin
#     # Test 1: Gdy bias=true, funkcja powinna zwrócić macierz zerową o podanych wymiarach
#     weights = Variable(rand(Float32, 3, 5))  # Przykładowa macierz wag
#     result = create_bias(weights, true)
#     @test result isa Variable
#     @test size(result.output) == (3,)
#     @test all(x -> x == 0, result.output)
    
#     # Test 2: Gdy bias=false, funkcja powinna zwrócić false
#     result = create_bias(weights, false)
#     @test result === nothing
    
#     # Test 3: Sprawdzenie, czy typ macierzy wynikowej jest zgodny z typem macierzy wag
#     weights_double = Variable(rand(Float64, 3, 5))
#     result = create_bias(weights_double, true)
#     @test eltype(result) == eltype(weights_double)
    
#     # Test 4: Sprawdzenie dla większej liczby wymiarów
#     result = create_bias(weights, true)
#     @test size(result) == (2, 3)
#     @test all(x -> x == 0, result)
    
#     # Test 5: Sprawdzenie działania z pustymi wymiarami
#     result = create_bias(weights, true)
#     @test size(result) == ()
    
#     # Test 6: Sprawdzenie w kontekście konstruktora Dense
#     weights = Variable(rand(Float32, 4, 6))
#     dense_with_bias = Dense(weights, true, identity)
#     @test dense_with_bias.biases isa Matrix{Float32}
#     @test size(dense_with_bias.biases) == (4, 1)
#     @test all(x -> x == 0, dense_with_bias.biases)
    
#     dense_without_bias = Dense(weights, false, identity)
#     @test dense_without_bias.biases === nothing
# end

# @testset "Chain Tests" begin
#     @testset "Empty Chain" begin
#         Chain = Chain()
#         x = 5
#         @test Chain(x) == x
#     end

#     @testset "Single Layer" begin
#         f = x -> x + 1
#         Chain = Chain(f)
#         x = 5
#         @test Chain(x) == f(x)
#     end

#     @testset "Multiple Layers" begin
#         f = x -> x + 1
#         g = x -> x * 2
#         h = x -> x - 3
#         Chain = Chain(f, g, h)
#         x = 5
#         expected = h(g(f(x)))
#         @test Chain(x) == expected
#     end

#     @testset "Different Layer Types" begin
#         f = x -> x .+ 1  # Operacja na wektorze
#         g = x -> sum(x)  # Redukcja do skalara
#         Chain = Chain(f, g)
#         x = [1, 2, 3]
#         expected = g(f(x))
#         @test Chain(x) == expected
#     end

#     @testset "Input and Output Types" begin
#         f = x -> string(x)
#         g = s -> length(s)
#         Chain = Chain(f, g)
#         x = 123
#         expected = g(f(x))
#         @test Chain(x) == expected
#     end
# end

# @testset "Graph and Operators" begin
#     @testset "GraphNode and Variable" begin
#         # Test Variable constructor
#         v = Variable([1.0, 2.0, 3.0], name="test_var")
#         @test v.output == [1.0, 2.0, 3.0]
#         @test v.name == "test_var"
#         @test v.∇ === nothing
        
#         # Test Variable constructor with default name
#         v2 = Variable([4.0, 5.0, 6.0])
#         @test v2.output == [4.0, 5.0, 6.0]
#         @test v2.name == "?"
#     end
    
#     @testset "Operators" begin
#         # Test ScalarOperator
#         v1 = Variable([1.0, 2.0], name="input1")
#         v2 = Variable([3.0, 4.0], name="input2")
        
#         op = ScalarOperator(+, v1, v2, name="addition")
#         @test op.inputs == (v1, v2)
#         @test op.name == "addition"
#         @test op.output === nothing
#         @test op.∇ === nothing
        
#         # Test BroadcastedOperator
#         bop = BroadcastedOperator(*, v1, v2, name="element_mul")
#         @test bop.inputs == (v1, v2)
#         @test bop.name == "element_mul"
#         @test bop.output === nothing
#         @test bop.∇ === nothing
#     end
    
#     @testset "Topological Sort" begin
#         # Create a simple computational graph
#         v1 = Variable([1.0, 2.0], name="input1")
#         v2 = Variable([3.0, 4.0], name="input2")
#         op1 = ScalarOperator(+, v1, v2, name="add")
#         v3 = Variable([5.0, 6.0], name="input3")
#         op2 = ScalarOperator(*, op1, v3, name="mul")
        
#         # Test topological sort
#         sorted = topological_sort(op2)
        
#         # Verify nodes appear in correct order (dependencies before dependents)
#         v1_idx = findfirst(x -> x === v1, sorted)
#         v2_idx = findfirst(x -> x === v2, sorted)
#         op1_idx = findfirst(x -> x === op1, sorted)
#         v3_idx = findfirst(x -> x === v3, sorted)
#         op2_idx = findfirst(x -> x === op2, sorted)
        
#         @test v1_idx < op1_idx
#         @test v2_idx < op1_idx
#         @test op1_idx < op2_idx
#         @test v3_idx < op2_idx
#         @test op2_idx == length(sorted)  # Final node should be the head
#     end
# end

# doubleInput(x::GraphNode) = BroadcastedOperator(doubleInput, x)
# forward(::BroadcastedOperator{typeof(doubleInput)}, x) = x .* 2
# backward(::BroadcastedOperator{typeof(doubleInput)}, x, ∇) = 
#     tuple(∇ .* 2)

# @testset "Dense Layer" begin
#     @testset "Construction" begin
#         # Test Dense constructor with explicit weights
#         w = Variable(randn(3, 2), name="weights")
#         dense = Dense(w, true, tanh)
#         @test dense.weights === w
#         @test dense.bias !== nothing
#         @test dense.bias.output == zeros(3)
#         @test dense.activation === tanh
        
#         # Test Dense constructor without bias
#         dense_no_bias = Dense(w, false, σ)
#         @test dense_no_bias.weights === w
#         @test dense_no_bias.bias === nothing
#         @test dense_no_bias.activation === σ
        
#         # Test Dense constructor with dimensions
#         dense_dims = Dense(2 => 3, ReLU, name="test_layer")
#         @test size(dense_dims.weights.output) == (3, 2)
#         @test dense_dims.weights.name == "test_layer"
#         @test dense_dims.activation === ReLU
#     end

#     @testset "One Dense Pass x->x" begin
#         # Arrange
#         w = Variable([1.0 2.0; 3.0 4.0], name="weights")
#         model = Chain(
#             Dense(w, true, identity),
#         )

#         dense = model.layers[1]
#         dense.bias.output = [0.1, 0.2]
        
#         x = Variable([0.5, 0.6], name="input")

#         # Act
#         graph = model(x)
#         graph_sorted = topological_sort(graph)
#         forward!(graph_sorted)

#         output = graph

        
#         # Assert
#         expected = [0.1, 0.2] + [1.0 2.0; 3.0 4.0] * [0.5, 0.6]
        
#         @test output isa GraphNode
#         @test output.output == expected
#     end

#     @testset "One Dense Pass x->2x" begin
#         # Arrange
#         w = Variable([1.0 2.0; 3.0 4.0], name="weights")


#         model = Chain(
#             Dense(w, true, doubleInput)
#         )

#         dense = model.layers[1]
#         dense.bias.output = [0.1, 0.2]
        
#         x = Variable([0.5, 0.6], name="input")

#         # Act
#         graph = model(x)
#         graph_sorted = topological_sort(graph)
#         forward!(graph_sorted)
#         output = graph


#         # Assert
#         expected = 2*([0.1, 0.2] + [1.0 2.0; 3.0 4.0] * [0.5, 0.6])
        
#         @test output isa GraphNode
#         @test output.output == expected
#     end
# end

# @testset "Dense Chain" begin
#     @testset "Construction" begin
#         # Create layers
#         layer1 = Dense(2 => 3, tanh, name="hidden")
#         layer2 = Dense(3 => 1, σ, name="output")
        
#         # Create Chain
#         net = Chain(layer1, layer2)
        
#         @test net.layers isa Tuple
#         @test length(net.layers) == 2
#         @test net.layers[1] === layer1
#         @test net.layers[2] === layer2
#     end
    
#     @testset "Forward Pass" begin
#         layer1 = Dense(2 => 3, tanh, name="hidden")
#         layer2 = Dense(3 => 1, σ, name="output")
#         net = Chain(layer1, layer2)
        
#         layer1.weights.output = [0.1 0.2; 0.3 0.4; 0.5 0.6]
#         layer1.bias = Variable([0.01, 0.02, 0.03], name="bias1")
#         layer2.weights.output = [0.7 0.8 0.9]
#         layer2.bias = Variable([0.04], name="bias2")
        
#         x = Variable([1.0, 2.0], name="input")
        
#         # Act
#         graph = net(x)
#         graph_sorted = topological_sort(graph)
#         forward!(graph_sorted)

#         output = graph
        
#         # Assert
#         expected = 0.865
#         @test output isa GraphNode
#         @test isapprox(output.output..., expected, atol=1e-3)

#     end
# end

# @testset "Loss Function and Model Integration" begin
#     # Create a simple model
#     input_neurons = 2
#     hidden_neurons = 3
#     output_neurons = 2
    
#     # Define activation functions
#     σ = tanh
#     softmax = z -> let y = exp.(z); y ./ sum(y) end
    
#     model = Chain(
#         Dense(input_neurons => hidden_neurons, σ, name="x̂"),
#         Dense(hidden_neurons => output_neurons, softmax, name="ŷ"),
#     )
    
#     # Initialize weights and biases for deterministic testing
#     model.layers[1].weights.output = [0.1 0.2; 0.3 0.4; 0.5 0.6]
#     model.layers[1].bias = Variable([0.01, 0.02, 0.03], name="bias1")
#     model.layers[2].weights.output = [0.7 0.8 0.9; 1.0 1.1 1.2]
#     model.layers[2].bias = Variable([0.04, 0.05], name="bias2")
    
#     # Create input and target
#     x = Variable([1.0, 2.0], name="input")
#     y = Variable([0.0, 1.0], name="target")  # One-hot encoding
    
#     # Create input node
#     x_node = ScalarOperator(identity, x, name="input_node")
#     x_node.output = x.output
#     y_node = ScalarOperator(identity, y, name="target_node")
#     y_node.output = y.output
    
#     @testset "Model Forward Pass" begin
#         output = model(x_node)
#         @test output isa GraphNode
#         @test length(output.output) == output_neurons
#         @test isapprox(sum(output.output), 1.0, atol=1e-6)  # Softmax output sums to 1
#     end
    
#     @testset "Loss Calculation" begin
#         function loss(x, y, model)
#             ŷ = model(x)
#             E = cross_entropy_loss(y.output, ŷ.output)
#             E.name = "loss"
#             return E
#         end
        
#         loss_node = loss(x_node, y_node, model)
#         @test loss_node isa GraphNode
#         @test loss_node.name == "loss"
#     end
    
#     @testset "Graph Construction" begin
#         function loss(x, y, model)
#             ŷ = model(x)
#             E = cross_entropy_loss(y.output, ŷ.output)
#             E.name = "loss"
#             return E
#         end
        
#         graph = topological_sort(loss(x_node, y_node, model))
#         @test length(graph) > 0
#         @test graph[end].name == "loss"
#     end
# end

# @testset "Initialization Functions" begin
#     # Test glorot_uniform if it exists
#     # Assuming glorot_uniform is defined somewhere in the code
#     if @isdefined glorot_uniform
#         @testset "Glorot Uniform Initialization" begin
#             out, in = 10, 5
#             w = glorot_uniform(out, in)
#             @test size(w) == (out, in)
            
#             # Check variance is approximately correct for Glorot initialization
#             scale = sqrt(6 / (in + out))
#             @test -scale <= minimum(w) <= 0 || 0 <= maximum(w) <= scale
#             @test std(w) ≈ scale / sqrt(3) rtol=0.5  # Rough check for uniform distribution variance
#         end
#     else
#         # Define a minimal implementation for testing
#         glorot_uniform(out, in) = rand(out, in) .* 2 .* sqrt(6 / (in + out)) .- sqrt(6 / (in + out))
        
#         @testset "Glorot Uniform Implementation" begin
#             out, in = 10, 5
#             w = glorot_uniform(out, in)
#             @test size(w) == (out, in)
#         end
#     end
# end

# @testset "End-to-End Model Training" begin
#     # This test simulates a simple training loop for the neural Chain
    
#     # Define utility functions for testing
#     function update_weights!(model, learning_rate=0.01)
#         # Simple gradient descent update
#         for layer in model.layers
#             if layer isa Dense
#                 layer.weights.output .-= learning_rate .* layer.weights.∇
#                 if !isnothing(layer.bias)
#                     layer.bias.output .-= learning_rate .* layer.bias.∇
#                 end
#             end
#         end
#     end
    
#     function forward_and_backward!(graph)
#         # Forward pass
#         for node in graph
#             if node isa Operator
#                 # Mock forward computations for testing
#                 if node.name == "loss"
#                     node.output = 1.0  # Dummy loss value
#                     node.∇ = 1.0
#                 elseif node isa ScalarOperator || node isa BroadcastedOperator
#                     # Simple mock outputs
#                     node.output = rand(size(node.inputs[1].output)...)
#                     node.∇ = rand(size(node.output)...)
#                 end
#             end
#         end
        
#         # Backward pass (simplified for testing)
#         for node in reverse(graph)
#             if node isa Variable
#                 node.∇ = rand(size(node.output)...)
#             end
#         end
#     end
    
#     # Create a simple model for training
#     input_neurons = 2
#     hidden_neurons = 3
#     output_neurons = 2
    
#     model = Chain(
#         Dense(input_neurons => hidden_neurons, tanh, name="hidden"),
#         Dense(hidden_neurons => output_neurons, 
#               z -> let y = exp.(z); y ./ sum(y) end, 
#               name="output")
#     )
    
#     # Create mock data
#     x = Variable(rand(input_neurons), name="x")
#     y = Variable(Float64[0, 1], name="y")  # One-hot encoding
    
#     # Create input nodes
#     x_node = ScalarOperator(identity, x, name="x_node")
#     x_node.output = x.output
#     y_node = ScalarOperator(identity, y, name="y_node")
#     y_node.output = y.output
    
#     # Define loss function
#     function loss(x, y, model)
#         ŷ = model(x)
#         E = ScalarOperator(-, y_node, ŷ, name="loss")  # Dummy loss for testing
#         E.name = "loss"
#         return E
#     end
    
#     # Build computational graph
#     graph = topological_sort(loss(x_node, y_node, model))
    
#     # Run a few training iterations
#     n_epochs = 3
#     for epoch in 1:n_epochs
#         forward_and_backward!(graph)
#         update_weights!(model)
#     end
    
#     # No specific assertions here as we're just testing that the training loop runs
#     @test true
# end

@testset "Chain Tests" begin
    @testset "Empty Chain" begin
        Chain = Chain()
        x = 5
        @test Chain(x) == x
    end

    @testset "Single Layer" begin
        f = x -> x + 1
        Chain = Chain(f)
        x = 5
        @test Chain(x) == f(x)
    end

    @testset "Multiple Layers" begin
        f = x -> x + 1
        g = x -> x * 2
        h = x -> x - 3
        Chain = Chain(f, g, h)
        x = 5
        expected = h(g(f(x)))
        @test Chain(x) == expected
    end

    @testset "Different Layer Types" begin
        f = x -> x .+ 1  # Operation on vector
        g = x -> sum(x)  # Reduction to scalar
        Chain = Chain(f, g)
        x = [1, 2, 3]
        expected = g(f(x))
        @test Chain(x) == expected
    end

    @testset "Input and Output Types" begin
        f = x -> string(x)
        g = s -> length(s)
        Chain = Chain(f, g)
        x = 123
        expected = g(f(x))
        @test Chain(x) == expected
    end
end

@testset "Graph and Operators" begin
    @testset "GraphNode and Variable" begin
        # Test Variable constructor
        v = Variable([1.0, 2.0, 3.0], name="test_var")
        @test v.output == [1.0, 2.0, 3.0]
        @test v.name == "test_var"
        @test v.∇ === nothing
        
        # Test Variable constructor with default name
        v2 = Variable([4.0, 5.0, 6.0])
        @test v2.output == [4.0, 5.0, 6.0]
        @test v2.name == "?"
    end
    
    @testset "Operators" begin
        # Test ScalarOperator
        v1 = Variable([1.0, 2.0], name="input1")
        v2 = Variable([3.0, 4.0], name="input2")
        
        op = ScalarOperator(+, v1, v2, name="addition")
        @test op.inputs == (v1, v2)
        @test op.name == "addition"
        @test op.output === nothing
        @test op.∇ === nothing
        
        # Test BroadcastedOperator
        bop = BroadcastedOperator(*, v1, v2, name="element_mul")
        @test bop.inputs == (v1, v2)
        @test bop.name == "element_mul"
        @test bop.output === nothing
        @test bop.∇ === nothing
    end
    
    @testset "Topological Sort" begin
        # Create a simple computational graph
        v1 = Variable([1.0, 2.0], name="input1")
        v2 = Variable([3.0, 4.0], name="input2")
        op1 = ScalarOperator(+, v1, v2, name="add")
        v3 = Variable([5.0, 6.0], name="input3")
        op2 = ScalarOperator(*, op1, v3, name="mul")
        
        # Test topological sort
        sorted = topological_sort(op2)
        
        # Verify nodes appear in correct order (dependencies before dependents)
        v1_idx = findfirst(x -> x === v1, sorted)
        v2_idx = findfirst(x -> x === v2, sorted)
        op1_idx = findfirst(x -> x === op1, sorted)
        v3_idx = findfirst(x -> x === v3, sorted)
        op2_idx = findfirst(x -> x === op2, sorted)
        
        @test v1_idx < op1_idx
        @test v2_idx < op1_idx
        @test op1_idx < op2_idx
        @test v3_idx < op2_idx
        @test op2_idx == length(sorted)  # Final node should be the head
    end
end

# Define activation functions for tests
doubleInput(x::GraphNode) = BroadcastedOperator(doubleInput, x)
forward(::BroadcastedOperator{typeof(doubleInput)}, x) = x .* 2
backward(::BroadcastedOperator{typeof(doubleInput)}, x, ∇) = 
    tuple(∇ .* 2)

@testset "Dense Layer" begin
    @testset "Construction" begin
        # Test Dense constructor with explicit weights
        w = Variable(randn(3, 2), name="weights")
        dense = Dense(w, true, tanh)
        @test dense.weights === w
        @test dense.bias !== nothing
        @test dense.bias.output == zeros(3)
        @test dense.activation === tanh
        
        # Test Dense constructor without bias
        dense_no_bias = Dense(w, false, σ)
        @test dense_no_bias.weights === w
        @test dense_no_bias.bias === nothing
        @test dense_no_bias.activation === σ
        
        # Test Dense constructor with dimensions
        dense_dims = Dense(2 => 3, ReLU, name="test_layer")
        @test size(dense_dims.weights.output) == (3, 2)
        @test dense_dims.weights.name == "test_layer"
        @test dense_dims.activation === ReLU
    end

    @testset "One Dense Pass x->x" begin
        # Arrange
        w = Variable([1.0 2.0; 3.0 4.0], name="weights")
        model = Chain(
            Dense(w, true, identity),
        )

        dense = model.layers[1]
        dense.bias.output = [0.1, 0.2]
        
        x = Variable([0.5, 0.6], name="input")

        # Act
        graph = model(x)
        graph_sorted = topological_sort(graph)
        forward!(graph_sorted)

        output = graph
        
        # Assert
        expected = [0.1, 0.2] + [1.0 2.0; 3.0 4.0] * [0.5, 0.6]
        
        @test output isa GraphNode
        @test output.output == expected
    end

    @testset "One Dense Pass x->2x" begin
        # Arrange
        w = Variable([1.0 2.0; 3.0 4.0], name="weights")

        model = Chain(
            Dense(w, true, doubleInput)
        )

        dense = model.layers[1]
        dense.bias.output = [0.1, 0.2]
        
        x = Variable([0.5, 0.6], name="input")

        # Act
        graph = model(x)
        graph_sorted = topological_sort(graph)
        forward!(graph_sorted)
        output = graph

        # Assert
        expected = 2*([0.1, 0.2] + [1.0 2.0; 3.0 4.0] * [0.5, 0.6])
        
        @test output isa GraphNode
        @test output.output == expected
    end
end

@testset "Dense Chain" begin
    @testset "Construction" begin
        # Create layers
        layer1 = Dense(2 => 3, tanh, name="hidden")
        layer2 = Dense(3 => 1, σ, name="output")
        
        # Create Chain
        net = Chain(layer1, layer2)
        
        @test net.layers isa Tuple
        @test length(net.layers) == 2
        @test net.layers[1] === layer1
        @test net.layers[2] === layer2
    end
    
    @testset "Forward Pass" begin
        layer1 = Dense(2 => 3, tanh, name="hidden")
        layer2 = Dense(3 => 1, σ, name="output")
        net = Chain(layer1, layer2)
        
        layer1.weights.output = [0.1 0.2; 0.3 0.4; 0.5 0.6]
        layer1.bias = Variable([0.01, 0.02, 0.03], name="bias1")
        layer2.weights.output = [0.7 0.8 0.9]
        layer2.bias = Variable([0.04], name="bias2")
        
        x = Variable([1.0, 2.0], name="input")
        
        # Act
        graph = net(x)
        graph_sorted = topological_sort(graph)
        forward!(graph_sorted)

        output = graph
        
        # Assert
        expected = 0.865
        @test output isa GraphNode
        @test isapprox(output.output..., expected, atol=1e-3)
    end
end

@testset "Activation Functions" begin
    @testset "tanh" begin
        x = Variable(0.5, name="x")
        node = tanh.(x)
        
        @test node isa GraphNode
        @test node isa BroadcastedOperator{typeof(tanh)}
        
        sorted = topological_sort(node)
        forward!(sorted)
        
        expected = tanh(0.5)
        @test isapprox(node.output, expected)
        
        backward!(sorted)
        
        expected_grad = 1.0 - tanh(0.5)^2
        @test isapprox(x.∇, expected_grad)
    end
    
    @testset "sigmoid (σ)" begin
        x = Variable(2.0, name="x")
        node = σ(x)
        
        @test node isa GraphNode
        @test node isa BroadcastedOperator{typeof(σ)}
        
        sorted = topological_sort(node)
        forward!(sorted)
        
        expected = 1.0 / (1.0 + exp(-2.0))
        @test isapprox(node.output, expected)
        
        backward!(sorted)
        
        expected_grad = expected * (1.0 - expected)
        @test isapprox(x.∇, expected_grad)
    end
    
    @testset "softmax" begin       
        x = Variable(-0.7, name="x")
        node = softmax(x)
        
        @test node isa GraphNode
        @test node isa BroadcastedOperator{typeof(softmax)}
        
        sorted = topological_sort(node)
        forward!(sorted)
        
        @test isapprox(node.output, 1.0)
        
        backward!(sorted)
        
        @test isapprox(x.∇, 0.0)
    end
end

@testset "Loss Functions" begin
    @testset "Mean Squared Error" begin      
        y_true = Variable([1.0], name="y_true")
        y_pred = Variable([0.9], name="y_pred")
        
        loss_node = mse_loss(y_true, y_pred)
        sorted = topological_sort(loss_node)
        forward!(sorted)
        
        expected_loss = 0.5 * (1.0 - 0.9)^2
        @test isapprox(loss_node.output..., expected_loss)

        backward!(sorted)

        expected_grad_pred = [0.9 - 1.0]
        @test isapprox(y_pred.∇, expected_grad_pred)
    end
    
    @testset "Cross Entropy Loss" begin
        y_true = Variable([0.0f0, 1.0f0], name="y_true")
        y_pred = Variable([0.1f0, 0.9f0], name="y_pred")
        
        loss_node = cross_entropy_loss(y_true, y_pred)
        sorted = topological_sort(loss_node)
        forward!(sorted)
        
        ϵ = eps(Float32)
        expected_loss = -sum(y_true.output .* log.(y_pred.output .+ ϵ))
        @test isapprox(loss_node.output, expected_loss)

        backward!(sorted)

        expected_grad_pred = Float32[0.0f0, -1.0f0/(0.9f0 + ϵ)]
        @test isapprox(y_pred.∇, expected_grad_pred)
    end
end

@testset "Loss Function and Model Integration" begin
    input_neurons = 2
    hidden_neurons = 3
    output_neurons = 2
     
    model = Chain(
        Dense(input_neurons => hidden_neurons, σ, name="hidden"),
        Dense(hidden_neurons => output_neurons, softmax, name="output"),
    )
    
    model.layers[1].weights.output = [0.1 0.2; 0.3 0.4; 0.5 0.6]
    model.layers[1].bias = Variable([0.01, 0.02, 0.03], name="bias1")
    model.layers[2].weights.output = [0.7 0.8 0.9; 1.0 1.1 1.2]
    model.layers[2].bias = Variable([0.04, 0.05], name="bias2")
    
    x = Variable([1.0, 2.0], name="input")
    y = Variable([0.0, 1.0], name="target") 
    
    @testset "LFMI: Model Forward Pass" begin
        output = model(x)
        sorted = topological_sort(output)
        forward!(sorted)
        
        @test output isa GraphNode
        @test length(output.output) == output_neurons
        @test isapprox(sum(output.output), 1.0, atol=1e-3) 
        
        h = tanh.([0.01, 0.02, 0.03] + [0.1 0.2; 0.3 0.4; 0.5 0.6] * [1.0, 2.0])
        z = [0.04, 0.05] + [0.7 0.8 0.9; 1.0 1.1 1.2] * h
        expected = exp.(z) ./ sum(exp.(z))
        
        @test isapprox(output.output, expected, atol=1e-2)
    end
    
    @testset "LFMI: Loss Calculation" begin
        function loss(x, y, model)
            ŷ = model(x)
            return cross_entropy_loss(y, ŷ)
        end
        
        loss_node = loss(x, y, model)
        sorted = topological_sort(loss_node)
        forward!(sorted)
        
        @test loss_node isa GraphNode
        
        h = tanh.([0.01, 0.02, 0.03] + [0.1 0.2; 0.3 0.4; 0.5 0.6] * [1.0, 2.0])
        z = [0.04, 0.05] + [0.7 0.8 0.9; 1.0 1.1 1.2] * h
        ŷ_expected = exp.(z) ./ sum(exp.(z))
        expected_loss = -sum([0.0, 1.0] .* log.(ŷ_expected .+ 1e-10))
        
        @test isapprox(loss_node.output, expected_loss, atol=1e-2)
    end
    
    @testset "LFMI: Backward Pass" begin
        function loss(x, y, model)
            ŷ = model(x)
            return cross_entropy_loss(y, ŷ)
        end
        
        loss_node = loss(x, y, model)
        sorted = topological_sort(loss_node)
        forward!(sorted)
                
        backward!(sorted)
        
        @test model.layers[1].weights.∇ !== nothing
        @test model.layers[1].bias.∇ !== nothing
        @test model.layers[2].weights.∇ !== nothing
        @test model.layers[2].bias.∇ !== nothing
    end
end