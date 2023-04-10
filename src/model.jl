import StatsBase: sample, mean
import ChainRulesCore: @ignore_derivatives
import Random: randn!
import CUDA
import Flux
import Flux: Dense, Parallel, Chain, LayerNorm, BatchNorm, Upsample, SamePad, leakyrelu, gradient, sigmoid
import Flux.Losses: logitbinarycrossentropy
import FluxTraining
import FastAI: ToDevice, handle
import Metalhead
import LinearAlgebra: norm

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
kl_divergence(μ, logσ²) = sum(@. ((μ^2 + exp(logσ²) - 1 - logσ²) / 2); dims=1)
function ELBO(x, x̄, μ, logσ²; warmup_factor::Rational=1//1)
  reconstruction_error = bernoulli_loss(x, x̄)
  kl_divergence_error = kl_divergence(μ, logσ²)
  return mean(reconstruction_error) + warmup_factor*mean(kl_divergence_error)
end
# We need this for FluxTraining.fit!
ELBO((x, x̄, μ, logσ²)::Tuple; warmup_factor::Rational=1//1) = ELBO(x, x̄, μ, logσ²;
                                                                   warmup_factor=warmup_factor)
ELBO((x̄, μ, logσ²)::Tuple, x; warmup_factor::Rational=1//1) = ELBO(x, x̄, μ, logσ²;
                                                                   warmup_factor=warmup_factor)
reg_l2(params) = sum(x->sum(x.^2), params)
########################

#### Set up model #########
# image size is (64, 64)
# backbone_dim = 512
# backbone_dim = 768
# latent_dim = 64

resnet_backbone() = let backbone = Metalhead.ResNet(18; pretrain=true)
   Chain(backbone.layers[1], Chain(backbone.layers[2].layers[1:2]..., identity))
end
# convnext_backbone() = Metalhead.ConvNeXt(:tiny; nclasses=backbone_dim)
convnext_backbone() = let backbone = Metalhead.ConvNeXt(:tiny)
    # 768 output dimension by default
    Chain(backbone.layers[1], Chain(backbone.layers[2].layers[[1, 2]]...))
end
backbone() = convnext_backbone()

bridge(backbone_dim, latent_dim) = Chain(
          Dense(backbone_dim, 128, leakyrelu),
          LayerNorm(128),
          Parallel(
              tuple,
              Dense(1//10*Flux.glorot_uniform(latent_dim, 128),
                    zeros(Float32, latent_dim)),  # mu
              # Special initialization, see https://arxiv.org/pdf/2010.14407.pdf, Table 2 (Appendix)
              Dense(1//10*Flux.glorot_uniform(latent_dim, 128),
                    -1*ones(Float32, latent_dim)),  # logvar
            )
        )


decoder() = Chain(Dense(latent_dim, 4*4*16, leakyrelu),
                Dense(4*4*16, 4*4*64, identity),
                xs -> reshape(xs, 4, 4, 64, :),
                ResidualBlock(64),
                ResidualBlock(64),
                Upsample(2),
                ResidualBlock(64),
                ResidualBlock(64),
                Conv((1, 1), 64=>32, identity),
                ResidualBlock(32),
                ResidualBlock(32),
                Upsample(2),
                Conv((1, 1), 32=>16, identity),
                Upsample(2),
                leakyrelu,
                Conv((5, 5), 16=>3, identity; pad=SamePad(), stride=1),
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
  params = Flux.params(learner.model.encoder, learner.model.bridge, learner.model.decoder)
  FluxTraining.runstep(learner, phase, (; x_lhs=x_lhs, v_lhs=v_lhs, x_rhs=x_rhs, v_rhs=v_rhs, ks=ks)) do handle, state
    gs = gradient(params) do
      intermediate_lhs   = learner.model.encoder(state.x_lhs)
      intermediate_rhs   = learner.model.encoder(state.x_rhs)
      μ_lhs, logσ²_lhs   = learner.model.bridge(intermediate_lhs)
      μ_rhs, logσ²_rhs   = learner.model.bridge(intermediate_rhs)
      # averaging mask with 1s for all style variables (which we always average)
      ks = let sz = (size(μ_lhs, 1) - size(state.ks, 1), size(state.ks, 2))
          vcat(state.ks, 1 .+ 0 .* similar(state.ks, sz))
      end
      μ̂_lhs              = ks.*(μ_lhs+μ_rhs)./2 + (1 .- ks).*(μ_lhs)
      μ̂_rhs              = ks.*(μ_lhs+μ_rhs)./2 + (1 .- ks).*(μ_rhs)
      logσ̂²_lhs = ks.*(logσ²_lhs+logσ²_rhs)./2 + (1 .- ks).*(logσ²_lhs)
      logσ̂²_rhs = ks.*(logσ²_lhs+logσ²_rhs)./2 + (1 .- ks).*(logσ²_rhs)
      state.z_lhs        = sample_latent(μ̂_lhs, logσ̂²_lhs)
      state.z_rhs        = sample_latent(μ̂_rhs, logσ̂²_rhs)
      state.x̄_lhs        = learner.model.decoder(state.z_lhs)
      state.x̄_rhs        = learner.model.decoder(state.z_rhs)
      state.y            = (state.x_lhs, state.v_lhs,
                            state.x_rhs, state.v_rhs)
      state.ŷ            = (state.x̄_lhs, state.z_lhs,
                            state.x̄_rhs, state.z_rhs)

      current_step = learner.cbstate.history[phase].steps
      warmup_factor::Rational = min(current_step // 10_000, 1//1)

      handle(FluxTraining.LossBegin())
      state.loss = (learner.lossfn(state.x_lhs, state.x̄_lhs, μ̂_lhs, logσ̂²_lhs;
                                   warmup_factor=warmup_factor)
                  + learner.lossfn(state.x_rhs, state.x̄_rhs, μ̂_rhs, logσ̂²_rhs;
                                   warmup_factor=warmup_factor)
                  + 1f-1*(cov_loss(state.z_lhs) + cov_loss(state.z_rhs))
                  + 1f-3*reg_l2(Flux.params(learner.model.decoder))  # we add some regularization here :)
                    )

      handle(FluxTraining.BackwardBegin())
      return state.loss
    end
    if norm(gs) > 1f6
      @warn "\n Large gradient with norm" norm(gs)
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

function FluxTraining.fit!(learner, nepochs::Int, (trainiter, validiter)::Tuple)
    for i in 1:nepochs
        epoch!(learner, VAETrainingPhase(), trainiter)
        epoch!(learner, VAEValidationPhase(), validiter)
    end
end
############################################################
