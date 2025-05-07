include("../src/structs.jl")
include("../src/scalar_operators.jl")
include("../src/broadcast_operators.jl")
include("../src/graph.jl")
include("../src/forward.jl")
include("../src/backward.jl")
include("../src/data_loader.jl")
include("../src/nn_model.jl")

rosenbrock(x, y) = (Constant(1.0) .- x .* x) .+ Constant(100.0) .* (y .- x .* x) .* (y .- x .* x)

import Random: shuffle, shuffle!, seed!
seed!(0)

input_neurons = 4
hidden_neurons = 8
output_neurons = 3
x = Variable(zeros(input_neurons), name="x")
y = Variable(zeros(output_neurons), name="y")
η = 0.001f0
epochs = 900

model = Network(
    Dense(input_neurons => hidden_neurons, 
          σ,
       #   bias = true,
          name="x̂"),
    Dense(hidden_neurons => output_neurons, 
          z -> let y = exp.(z); y ./ sum(y) end,
      #    bias = true,
          name="ŷ"),
)

function loss(x, y, model)
    ŷ = model(x)
    E = cross_entropy_loss(y, ŷ); E.name = "loss"
    return E
end

include("../data/iris.jl")
test_size  = 10
train_size = 140
data_size  = train_size + test_size
train_set  = shuffle(1:data_size)[1:train_size]
test_set   = setdiff(1:data_size, train_set)

graph = topological_sort(loss(x, y, model))
for (i,n) in enumerate(graph)
    print(i, ". "); println(n)
end

function test(set)
    L = 0.0
    for j = set
      x.output .= inputs[j,:]
      y.output .= targets[j,:]
      L += forward!(graph)
    end
    return L / length(set) 
end

function train!(batch, model, graph, lr=0.01f0)
    j = first(batch)
    x.output .= inputs[j,:]
    y.output .= targets[j,:]
    forward!(graph)
    backward!(graph)

    grads = gradient(model)

    for j in Iterators.drop(batch, 1)
        x.output .= inputs[j,:]
        y.output .= targets[j,:]

        forward!(graph)
        backward!(graph)
        
        accumulate_gradients!(grads, gradient(model))
    end

    update_params!(model, lr; grads, batch_len=length(batch))
    forward!(graph)
    return nothing
end

for i=1:epochs
    shuffle!(train_set)
    train!(train_set[1:10], model, graph, η)
    test(test_set)
end

x = Variable([0.], name="x")
y = Variable([0.], name="y")
graph = topological_sort(rosenbrock(x, y))


v  = -1:.1:+1
n  = length(v)
z  = zeros(n, n)
dz = zeros(n, n, 2)
for i=1:n, j=1:n
    x.output .= v[i]
    y.output .= v[j]
    z[i,j] = first(forward!(graph)); backward!(graph)
    dz[i,j,1] = first(x.∇)
    dz[i,j,2] = first(y.∇)
end

using PyPlot
xv = repeat(v, 1, n)
yv = repeat(v',n, 1)
contourf(xv, yv, z)
quiver(xv, yv, dz[:,:,1], dz[:,:,2])