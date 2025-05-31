# https://github.com/FluxML/Flux.jl/blob/aff2f9e5aaaed36282a02a4032c787b999513da2/src/data/dataloader.jl#L15-L76
# Data loader użyje wszystkich informacji - brak partial => zawsze includuje wszystkie dane
using Random: AbstractRNG, shuffle!, GLOBAL_RNG
using Base: @propagate_inbounds

struct DataLoader{D,R<:AbstractRNG}
    data::D
    num_obs::Int
    indicies_obs::Vector{Int}
    batchsize::Int
    num_batches::Int
    shuffle::Bool
    rng::R
    partial::Bool
end

function DataLoader(data; batchsize=1, shuffle=false, rng=GLOBAL_RNG, partial=false)
    if partial
        @warn "TODO: last mini batch"
    end 
    batchsize > 0 || throw(ArgumentError("batchsize has to be at least 1"))

    num_obs = _get_num_obs(data)
    if num_obs < batchsize
        @warn "number of observations less than batchsize, decreasing the batchsize to $num_obs"
        batchsize = num_obs
    end

    num_batches = partial ? ceil(Int, num_obs / batchsize) : floor(Int, num_obs / batchsize)
    
    indicies_obs = collect(1:num_obs)
    
    DataLoader(data, num_obs, indicies_obs, batchsize, num_batches, shuffle, rng, partial)
end

# returns data in d.indices[i+1:i+batchsize]
@propagate_inbounds function Base.iterate(dl::DataLoader, i=0) 
    if dl.partial
        i >= dl.num_obs && return nothing
    else
        i >= dl.num_batches * dl.batchsize && return nothing
    end

    # i >= dl.num_obs && return nothing
    if dl.shuffle && i == 0
        shuffle!(dl.rng, dl.indicies_obs)
    end

    next_idx = min(i + dl.batchsize, dl.num_obs)
    ids = dl.indicies_obs[i+1:next_idx]
    batch = _get_obs(dl.data, ids)
    return (batch, next_idx)
end

Base.length(dl::DataLoader) = dl.num_batches

_get_num_obs(data::AbstractArray) = size(data)[end]
function _get_num_obs(data::Union{Tuple, NamedTuple})
    isempty(data) && throw(ArgumentError("data container cannot be empty"))

    first_key = 1
    n_obs = _get_num_obs(data[first_key])
    _check_dim_consistency!(data, n_obs, first_key)
    return n_obs
end


_get_obs(data::AbstractArray, i) = data[ntuple(i -> Colon(), Val(ndims(data) - 1))..., i]
_get_obs(data::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(_get_obs, i), data)

function _check_dim_consistency!(data, n_obs::Int, first_key)
    for key in keys(data)
        current_obs = _get_num_obs(data[key])
        if current_obs != n_obs
            throw(DimensionMismatch(
                "data dimension mismatch:\n" *
                "- First element ($(repr(first_key))): $(summary(data[first_key])) has $n_obs observations\n" *
                "- Current element ($(repr(key))): $(summary(data[key])) has $current_obs observations\n" *
                "All elements must have consistent observation counts."
            ))
        end
    end
end