using Test
include("../src/nn_model.jl")

# @testset "create_bias tests" begin
#     # Test 1: Gdy bias=true, funkcja powinna zwrócić macierz zerową o podanych wymiarach
#     weights = rand(Float32, 3, 5)  # Przykładowa macierz wag
#     result = create_bias(weights, true, 3)
#     @test result isa Matrix{Float32}
#     @test size(result) == (3, 1)
#     @test all(x -> x == 0, result)
    
#     # Test 2: Gdy bias=false, funkcja powinna zwrócić false
#     result = create_bias(weights, false, 3)
#     @test result === false
    
#     # Test 3: Sprawdzenie, czy typ macierzy wynikowej jest zgodny z typem macierzy wag
#     weights_double = rand(Float64, 3, 5)
#     result = create_bias(weights_double, true, 3)
#     @test eltype(result) == eltype(weights_double)
    
#     # Test 4: Sprawdzenie dla większej liczby wymiarów
#     result = create_bias(weights, true, 2, 3)
#     @test size(result) == (2, 3)
#     @test all(x -> x == 0, result)
    
#     # Test 5: Sprawdzenie działania z pustymi wymiarami
#     result = create_bias(weights, true)
#     @test size(result) == ()
    
#     # Test 6: Sprawdzenie w kontekście konstruktora Dense
#     weights = rand(Float32, 4, 6)
#     dense_with_bias = Dense(weights, true, identity)
#     @test dense_with_bias.biases isa Matrix{Float32}
#     @test size(dense_with_bias.biases) == (4, 1)
#     @test all(x -> x == 0, dense_with_bias.biases)
    
#     dense_without_bias = Dense(weights, false, identity)
#     @test dense_without_bias.biases === false
# end

@testset "Network Tests" begin
    @testset "Empty Network" begin
        network = Network()
        x = 5
        @test network(x) == x
    end

    @testset "Single Layer" begin
        f = x -> x + 1
        network = Network(f)
        x = 5
        @test network(x) == f(x)
    end

    @testset "Multiple Layers" begin
        f = x -> x + 1
        g = x -> x * 2
        h = x -> x - 3
        network = Network(f, g, h)
        x = 5
        expected = h(g(f(x)))
        @test network(x) == expected
    end

    @testset "Different Layer Types" begin
        f = x -> x .+ 1  # Operacja na wektorze
        g = x -> sum(x)  # Redukcja do skalara
        network = Network(f, g)
        x = [1, 2, 3]
        expected = g(f(x))
        @test network(x) == expected
    end

    @testset "Input and Output Types" begin
        f = x -> string(x)
        g = s -> length(s)
        network = Network(f, g)
        x = 123
        expected = g(f(x))
        @test network(x) == expected
    end
end