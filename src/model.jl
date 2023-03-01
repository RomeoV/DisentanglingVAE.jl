import StatsBase: sample, mean
import ChainRulesCore: @ignore_derivatives
import Random: randn!
import CUDA
using Flux
import Flux: Chain
import Flux.Losses: logitbinarycrossentropy
import FluxTraining
import FastAI: ToDevice, handle
import Metalhead

##### Set up VAE ##########
struct VAE{E, B, D}
  encoder::E
  bridge ::B
  decoder::D
end
Flux.@functor VAE
VAE(encoder, bridge, decoder, device) = VAE(encoder |> device,
                                            bridge |> device,
                                            decoder |> device)

function (vae::VAE)(x::AbstractArray{<:Real, 4})
  intermediate = vae.encoder(x)
  μ, logσ² = vae.bridge(intermediate)
  z = sample_latent(μ, logσ²)
  x̄ = vae.decoder(z)
  return x̄, μ, logσ²
end
function (vae::VAE)((x_lhs, v_lhs, x_rhs, v_rhs)::Tuple{<:AbstractArray{<:Real, 4},
                                                        <:AbstractArray{<:Real, 2},
                                                        <:AbstractArray{<:Real, 4},
                                                        <:AbstractArray{<:Real, 2},
                                                        <:AbstractArray{<:Real, 2}};
                   apply_sigmoid=false)
  intermediate_lhs = vae.encoder(x_lhs)
  intermediate_rhs = vae.encoder(x_rhs)
  μ_lhs, logσ²_lhs = vae.bridge(intermediate_lhs)
  μ_rhs, logσ²_rhs = vae.bridge(intermediate_rhs)
  z_lhs = sample_latent(μ_lhs, logσ²_lhs)
  z_rhs = sample_latent(μ_rhs, logσ²_rhs)
  x̄_lhs = vae.decoder(z_lhs)
  x̄_rhs = vae.decoder(z_rhs)
  T = apply_sigmoid ? sigmoid : identity
  return T.(x̄_lhs), z_lhs, T.(x̄_rhs), z_rhs
end

sample_latent(μ::AbstractArray{T}, logσ²::AbstractArray{T}) where {T} =
       μ .+ exp.(logσ²./2) .* randn!(similar(logσ²))

bernoulli_loss(x, logit_rec) = logitbinarycrossentropy(logit_rec, x;
                                                       agg=x->sum(x; dims=[1,2,3]))
function ELBO(x, x̄, μ, logσ²)
  # reconstruction_error = mean(sum(@. ((x̄ - x)^2); dims=(1,2,3)))
  reconstruction_error = bernoulli_loss(x, x̄)
  kl_divergence = sum(@. ((μ^2 + exp(logσ²) - 1 - logσ²) / 2); dims=1)
  return mean(reconstruction_error) + mean(kl_divergence)
end
# We need this for FluxTraining.fit!
ELBO((x, x̄, μ, logσ²)::Tuple) = ELBO(x, x̄, μ, logσ²)
########################

#### Set up model #########
# image size is (64, 64)
backbone_dim = 512
# latent_dim = 64

backbone() = let backbone = Metalhead.ResNet(18; pretrain=true)
   Chain(backbone.layers[1], Chain(backbone.layers[2].layers[1:2]..., identity))
 end

bridge(latent_dim) =
Chain(Dense(backbone_dim, backbone_dim, leakyrelu),
      LayerNorm(backbone_dim),
      Parallel(
          tuple,
          # Special initialization, see https://arxiv.org/pdf/2010.14407.pdf, Table 2 (Appendix)
          Dense(0.1f0*Flux.glorot_uniform(latent_dim, backbone_dim),
                -1*ones(Float32, latent_dim)),  # μ
          Dense(0.1f0*Flux.glorot_uniform(latent_dim, backbone_dim),
                -1*ones(Float32, latent_dim)),  # logvar
         )
     )

ResidualBlock(c) = Parallel(+,
                            Chain(leakyrelu,
                                  Conv((3, 3), c=>c, identity; pad=SamePad()), 
                                  leakyrelu,
                                  Conv((3, 3), c=>c, identity; pad=SamePad())),
                            identity)

decoder() = Chain(Dense(latent_dim, 4*4*64, leakyrelu),
                Dense(4*4*64, 4*4*256÷2, identity),
                xs -> reshape(xs, 4, 4, 256÷2, :),
                ResidualBlock(256÷2),
                ResidualBlock(256÷2),
                Upsample(2),
                ResidualBlock(256÷2),
                ResidualBlock(256÷2),
                Conv((1, 1), 256÷2=>128÷2, identity),
                ResidualBlock(128÷2),
                ResidualBlock(128÷2),
                Upsample(2),
                ResidualBlock(128÷2),
                ResidualBlock(128÷2),
                Conv((1, 1), 128÷2=>64÷2, identity),
                Upsample(2),
                leakyrelu,
                Conv((5, 5), 64÷2=>3; pad=SamePad()),
                xs -> xs[3:30, 3:30, :, :])

struct VAETrainingPhase <: FluxTraining.AbstractTrainingPhase end
struct VAEValidationPhase <: FluxTraining.AbstractValidationPhase end
function FluxTraining.on(
    ::FluxTraining.StepBegin,
    ::Union{VAETrainingPhase, VAEValidationPhase},
    cb::ToDevice,
    learner,
  )
  learner.step.x_lhs = cb.movedatafn(learner.step.x_lhs)
  learner.step.v_lhs = cb.movedatafn(learner.step.v_lhs)
  learner.step.x_rhs = cb.movedatafn(learner.step.x_rhs)
  learner.step.v_rhs = cb.movedatafn(learner.step.v_rhs)
  learner.step.ks    = begin T=eltype(learner.step.x_lhs);
    cb.movedatafn(convert.(T, learner.step.ks))
  end
end

function FluxTraining.step!(learner, phase::VAETrainingPhase, batch)
  (x_lhs, v_lhs, x_rhs, v_rhs, ks) = batch
  params = Flux.params(learner.model.bridge, learner.model.decoder)
  FluxTraining.runstep(learner, phase, (; x_lhs=x_lhs, v_lhs=v_lhs, x_rhs=x_rhs, v_rhs=v_rhs, ks=ks)) do handle, state
    intermediate_lhs   = @ignore_derivatives learner.model.encoder(state.x_lhs)
    intermediate_rhs   = @ignore_derivatives learner.model.encoder(state.x_rhs)
    gs = gradient(params) do
      μ_lhs, logσ²_lhs   = learner.model.bridge(intermediate_lhs)
      μ_rhs, logσ²_rhs   = learner.model.bridge(intermediate_rhs)
      μ_lhs = state.ks.*(μ_lhs+μ_rhs)./2 + (1 .- state.ks).*(μ_lhs)
      μ_rhs = state.ks.*(μ_lhs+μ_rhs)./2 + (1 .- state.ks).*(μ_rhs)
      state.z_lhs        = sample_latent(μ_lhs, logσ²_lhs)
      state.z_rhs        = sample_latent(μ_rhs, logσ²_rhs)
      state.x̄_lhs        = learner.model.decoder(state.z_lhs)
      state.x̄_rhs        = learner.model.decoder(state.z_rhs)
      state.y            = (state.x_lhs, state.v_lhs,
                            state.x_rhs, state.v_rhs)
      state.ŷ            = (state.x̄_lhs, state.z_lhs,
                            state.x̄_rhs, state.z_rhs)

      handle(FluxTraining.LossBegin())
      state.loss = (learner.lossfn(state.x_lhs, state.x̄_lhs, μ_lhs, logσ²_lhs)
                  + learner.lossfn(state.x_rhs, state.x̄_rhs, μ_rhs, logσ²_rhs))

      handle(FluxTraining.BackwardBegin())
      return state.loss
    end
    handle(FluxTraining.BackwardEnd())
    Flux.Optimise.update!(learner.optimizer, params, gs)
  end
end

function FluxTraining.step!(learner, phase::VAEValidationPhase, batch)
  (x_lhs, v_lhs, x_rhs, v_rhs, ks) = batch
  FluxTraining.runstep(learner, phase, (; x_lhs=x_lhs, v_lhs=v_lhs, x_rhs=x_rhs, v_rhs=v_rhs, ks=ks)) do handle, state
    @ignore_derivatives begin
      state.loss = ( learner.lossfn(state.x_lhs, learner.model(state.x_lhs)...)
                   + learner.lossfn(state.x_lhs, learner.model(state.x_lhs)...))
    end
  end
end

function FluxTraining.fit!(learner, nepochs::Int, (trainiter, validiter))
    for i in 1:nepochs
        epoch!(learner, VAETrainingPhase(), trainiter)
        epoch!(learner, VAEValidationPhase(), validiter)
    end
end
############################################################
