using Test
using Random
include("../src/data_loader.jl")

@testset "DataLoader" begin
    function create_dummy_data(n_samples=100, n_features=5)
        X = rand(Float32, n_features, n_samples)
        y = rand(0:1, n_samples)
        return X, y
    end
    
    @testset "Constructor" begin
        X, y = create_dummy_data()
        
        # Podstawowa konstrukcja
        dl = DataLoader((X, y), batchsize=10)
        @test dl.batchsize == 10
        @test dl.num_obs == 100
        @test dl.num_batches == 10
        @test dl.shuffle == false
        @test length(dl.indicies_obs) == 100
        
        # Test z tasowaniem
        dl_shuffled = DataLoader((X, y), batchsize=10, shuffle=true)
        @test dl_shuffled.shuffle == true
        @test sort(dl_shuffled.indicies_obs) == collect(1:100)
        @test dl_shuffled.indicies_obs != collect(1:100)  # Sprawdzenie czy indeksy zostały przetasowane
        
        # Test z własnym generatorem liczb losowych
        custom_rng = Random.MersenneTwister(42)
        dl_rng = DataLoader((X, y), batchsize=10, shuffle=true, rng=custom_rng)
        @test dl_rng.rng === custom_rng
        
        # Test gdy batchsize jest większy niż liczba próbek
        @test_logs (:warn, r"number of observations less than batchsize") DataLoader((X, y), batchsize=200)
        dl_auto_adjusted = DataLoader((X, y), batchsize=200)
        @test dl_auto_adjusted.batchsize == 100
        
        # Test błędu dla batchsize <= 0
        @test_throws ArgumentError DataLoader((X, y), batchsize=0)
        @test_throws ArgumentError DataLoader((X, y), batchsize=-5)
    end
    
    @testset "get_num_obs" begin
        # Dla macierzy
        X = rand(5, 100)
        @test _get_num_obs(X) == 100
        
        # Dla wielowymiarowych tablic
        X_3d = rand(3, 4, 50)
        @test _get_num_obs(X_3d) == 50
        
        # Dla krotek i nazwanych krotek
        X, y = create_dummy_data()
        @test _get_num_obs((X, y)) == 100
        
        named_data = (features=X, labels=y)
        @test _get_num_obs(named_data) == 100
        
        # Test dla pustej krotki
        @test_throws ArgumentError _get_num_obs(())
        
        # Test dla niespójnych wymiarów
        X_small = rand(5, 50)
        @test_throws DimensionMismatch _get_num_obs((X, X_small))
    end
    
    @testset "get_obs" begin
        X, y = create_dummy_data(100, 5)
        
        # Pojedyncza obserwacja
        @test size(_get_obs(X, 1)) == (5,)
        @test _get_obs(y, 1) == y[1]
        
        # Wiele obserwacji
        @test size(_get_obs(X, [1, 2, 3])) == (5, 3)
        @test _get_obs(y, [1, 2, 3]) == y[1:3]
        
        # Dla krotki
        batch_tuple = _get_obs((X, y), [1, 2, 3])
        @test size(batch_tuple[1]) == (5, 3)
        @test length(batch_tuple[2]) == 3
        
        # Dla nazwanej krotki
        named_data = (features=X, labels=y)
        batch_named = _get_obs(named_data, [1, 2, 3])
        @test size(batch_named.features) == (5, 3)
        @test length(batch_named.labels) == 3
    end
    
    @testset "Iteration" begin
        X, y = create_dummy_data(100, 5)
        
        # Test standardowej iteracji
        dl = DataLoader((X, y), batchsize=10)
        batches = collect(dl)
        @test dl.num_batches == 10
        @test size(batches[1][1]) == (5, 10)  # Rozmiar pierwszej partii
        @test length(batches[1][2]) == 10     # Rozmiar etykiet pierwszej partii
        
        # Test z mniejszym batchsize
        dl_small = DataLoader((X, y), batchsize=1)
        batches_small = collect(dl_small)
        @test dl_small.num_batches == 100
        @test size(batches_small[1][1]) == (5, 1)
        
        # Test z batchsize równym liczbie próbek
        dl_full = DataLoader((X, y), batchsize=100)
        batches_full = collect(dl_full)
        @test dl_full.num_batches == 1
        @test size(batches_full[1][1]) == (5, 100)
        
        # Test z tasowaniem
        dl_shuffled = DataLoader((X, y), batchsize=10, shuffle=true)
        batches_shuffled = collect(dl_shuffled)
        @test batches_shuffled[1][2] != batches[1][2]
    end
    
    @testset "Edge Cases" begin
        # Test dla jednowymiarowych danych
        X_1d = rand(100)
        @test _get_num_obs(X_1d) == 100
        
        # Test dla danych z jedną obserwacją
        X_single = rand(5, 1)
        y_single = [1]
        dl_single = DataLoader((X_single, y_single), batchsize=1)
        @test dl_single.num_batches == 1
        @test length(collect(dl_single)) == 1
        
        # Test dla danych z różnymi typami
        X_int = rand(Int, 5, 50)
        y_float = rand(Float32, 50)
        dl_mixed = DataLoader((X_int, y_float), batchsize=10)
        batch = first(dl_mixed)
        @test eltype(batch[1]) == Int
        @test eltype(batch[2]) == Float32
    end
end
nothing