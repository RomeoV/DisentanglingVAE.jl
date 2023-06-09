import Flux
import Flux: ofeltype, normalise
import MLUtils
import Random: randn!
import Flux.Statistics: mean, std
import Metalhead: ChannelLayerNorm

import Flux.Zygote.ChainRulesCore: @ignore_derivatives

@inline function Flux.normalise(x::AbstractArray{T, 2}; dims=ndims(x), ϵ=ofeltype(x, 1e-5)) where T<:Real
  μ = mean(x, dims=dims)
  noise = @ignore_derivatives ϵ*randn!(similar(x))
  σ = std(x.+noise, dims=dims, mean=μ, corrected=false)
  return @. (x - μ) / (σ + ϵ)
end


function (m::ChannelLayerNorm)(x::AbstractArray{T, 2}) where T<:Real
    m.diag(Flux.normalise(x; dims = ndims(x) - 1, ϵ = m.ϵ))
end
