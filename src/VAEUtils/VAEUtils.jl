module VAEUtils
import Random: randn!

export sample_latent

sample_latent(μ::AbstractArray{T, 2}, logσ²::AbstractArray{T, 2}) where {T} =
       μ .+ exp.(logσ²./2) .* randn!(similar(logσ²))

end
