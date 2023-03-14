import Distributions: quantile, Distribution, cdf
import DataStructures: DefaultOrderedDict
import StatsBase: mean, var
import IntervalSets: Interval

# From Romeo's thesis, Listing 3 in Chapter 4
function compute_calibration_values(predictions  :: Vector{<:Distribution},
                                   observations :: Vector{<:Real})
    ps = 0.05:0.05:0.95
    counts = DefaultOrderedDict{eltype(observations), Int}(0)
    for (pred, obs) in zip(predictions, observations)
        for p in ps
            counts[p] += obs ∈ invcdf_interval(pred, p)
        end
    end
    observed_frequencies = values(counts) ./ length(observations)
    return (ps, observed_frequencies)
end
function test_calibration(predictions  :: Vector{<:Distribution},
                          observations :: Vector{<:Real};
                          ϵ=0.10)
    ps, observed_frequencies = compute_calibration_values(predictions, observations)
    @assert all(observed_frequencies .>= (ps .* (1-ϵ)))
end
function compute_calibration_metric(predictions  :: Vector{<:Distribution},
                                    observations :: Vector{<:Real})
    ps, observed_frequencies = compute_calibration_values(predictions, observations)
    metric_vec = (observed_frequencies .- ps)
    return mean(metric_vec), minimum(metric_vec), maximum(metric_vec)
end

invcdf = quantile
function invcdf_interval(D::Distribution, p::Real)
    Interval(invcdf(D, 0.5-p/2), invcdf(D, 0.5+p/2))
end

function compute_dispersion(predictions  :: Vector{<:Distribution},
                            observations :: Vector{<:Real})
    var([cdf(pred, obs)
         for (pred, obs) in zip(predictions, observations)])
end
function test_dispersion(predictions  :: Vector{<:Distribution},
                         observations :: Vector{<:Real};
                         ϵ=0.10)
    @assert compute_dispersion(predictions, observations) <= 1/12 * (1+ϵ)
end
