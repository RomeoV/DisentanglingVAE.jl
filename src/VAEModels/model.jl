import Random: randn!
import Flux: Chain, LayerNorm
import ..VAEUtils: sample_latent

struct VAE{E, D}
  encoder::E
  decoder::D
end
Flux.@functor VAE
VAE() = VAE(Chain(ResidualEncoder(128; Block_t=ResidualBlock), LayerNorm(128), backbone_head(128=>6)),
            ResidualDecoder(6; Block_t=ResidualBlock))


function (vae::VAE)(x::AbstractArray{T, 4})::Tuple{AbstractArray{T, 4},
                                                   AbstractArray{T, 2},
                                                   AbstractArray{T, 2}} where T
  μ, logσ², escape = vae.encoder(x)
  z = sample_latent(μ, logσ²)  # + escape
  x̄ = vae.decoder(z)
  return x̄, μ, logσ²
end

function (vae::VAE)((x_lhs, x_rhs)::Tuple{<:AbstractArray{T, 4},
                                          <:AbstractArray{T, 4}}) where T
  return (vae(x_lhs)..., vae(x_rhs)...)
end

@testset "model eval" begin
  for device in [Flux.cpu, Flux.gpu]
    xs = rand(Float32, 32, 32, 3, 7) |> device
    model = VAE() |> device
    x̄, μ, logσ² = model(xs)
    @test all(isfinite, x̄)
    @test size(x̄) == (32, 32, 3, 7)
    @test all(isfinite, μ)
    @test size(μ) == (6, 7)
    @test all(isfinite, logσ²)
    @test size(logσ²) == (6, 7)
  end
end
