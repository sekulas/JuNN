module JuAD

# Export core types
export GraphNode, Operator, Constant, Variable
export ScalarOperator, BroadcastedOperator

# Export graph operations
export topological_sort, forward!, backward!
export reset!, compute!, update!

# Export utility functions
export visit

# Export operators
export forward, backward
export ReLU, sigmoid, σ, softmax, tanh
export *
export +, -, /, max, ^, exp, log, sum, mean
export getindex_col_batch, getindex_col


include("./structs.jl")
include("./graph.jl")
include("./forward.jl") 
include("./backward.jl")
include("./scalar_operators.jl")
include("./broadcast_operators.jl")
include("./pretty_printing.jl")

end