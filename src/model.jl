import StatsBase: sample, mean
import ChainRulesCore: @ignore_derivatives
import Random: randn!
import CUDA
import Flux
import Flux: Dense, Parallel, Chain, LayerNorm, BatchNorm, Upsample, SamePad, leakyrelu, gradient, sigmoid
import Flux.Losses: logitbinarycrossentropy
import FluxTraining
import FastAI
import FastAI: ToDevice, handle
import Metalhead
import LinearAlgebra: norm
import Match: @match
import MLUtils: batch

# model should support
# model(x)
# ŷ = model(x)
# lossfn(ŷ, y)
# We can use `checkblock(blocks.ŷ, model(x))`
# Recall that for the VAELearningTask, we have
# x = Tuple{x, v, x, v, k}
# x = Tuple{x, v, x, v, k}

##### Set up VAE ##########
struct VAE{E, D}
  encoder::E
  decoder::D
end
Flux.@functor VAE
# VAE(backbone, bridge, decoder) = VAE(Chain(backbone, bridge),
#                                      decoder)
VAE() = VAE(Chain(ResidualEncoder(128),  # <- output dim
                  bridge(128, 6)),
            ResidualDecoder(6))

AbstractImageTensor = AbstractArray{T, 4} where T
@kwdef struct VAEResultSingle{T<:AbstractImageTensor, M<:AbstractMatrix}
  x :: T
  v :: M
  k :: M
  μ :: M
  μ̂ :: M
  logσ² :: M
  logσ̂² :: M
  z :: M
  escape :: M
  x̄ :: T
end

struct VAEResultDouble{T, M}
  lhs :: VAEResultSingle{T, M}
  rhs :: VAEResultSingle{T, M}
end

function (vae::VAE)(x::AbstractImageTensor{T}) where T
  μ, logσ², escape = vae.encoder(x)
  z = sample_latent(μ, logσ²) + escape
  x̄ = vae.decoder(z)
  return (μ, logσ², escape), x̄
end

function (vae::VAE)((x_lhs, v_lhs, x_rhs, v_rhs, ks_c)::Tuple{<:AbstractImageTensor{T},
                                                              <:AbstractMatrix{T},
                                                              <:AbstractImageTensor{T},
                                                              <:AbstractMatrix{T},
                                                              <:AbstractMatrix{T}};
                   apply_sigmoid=false) where T <: Real
      activation = apply_sigmoid ? sigmoid : identity

      μ_lhs, logσ²_lhs, escape_lhs = vae.encoder(x_lhs)
      μ_rhs, logσ²_rhs, escape_rhs = vae.encoder(x_rhs)

      # averaging mask with 1s for all style variables (which we always average)
      ks_sc = let sz = (size(μ_lhs, 1) - size(ks_c, 1), size(ks_c, 2))
          @ignore_derivatives vcat(ks_c, 1 .+ 0 .* similar(ks_c, sz))
      end

      μ̂_lhs     = ks_sc.*(μ_lhs+μ_rhs)./2         + (1 .- ks_sc).*(μ_lhs)
      μ̂_rhs     = ks_sc.*(μ_lhs+μ_rhs)./2         + (1 .- ks_sc).*(μ_rhs)
      logσ̂²_lhs = ks_sc.*(logσ²_lhs+logσ²_rhs)./2 + (1 .- ks_sc).*(logσ²_lhs)
      logσ̂²_rhs = ks_sc.*(logσ²_lhs+logσ²_rhs)./2 + (1 .- ks_sc).*(logσ²_rhs)

      z_lhs     = sample_latent(μ̂_lhs, logσ̂²_lhs) + escape_lhs
      z_rhs     = sample_latent(μ̂_rhs, logσ̂²_rhs) + escape_rhs
      x̄_lhs     = activation.(vae.decoder(z_lhs))
      x̄_rhs     = activation.(vae.decoder(z_rhs))

      out_lhs = VAEResultSingle(
        x=x_lhs, v=v_lhs, k=ks_c, μ=μ_lhs, μ̂=μ̂_lhs, logσ²=logσ²_lhs,
        logσ̂²=logσ̂²_lhs, z=z_lhs, escape=escape_lhs, x̄=x̄_lhs)
      out_rhs = VAEResultSingle(
        x=x_rhs, v=v_rhs, k=ks_c, μ=μ_rhs, μ̂=μ̂_rhs, logσ²=logσ²_rhs,
        logσ̂²=logσ̂²_rhs, z=z_rhs, escape=escape_rhs, x̄=x̄_rhs)
      return VAEResultDouble(out_lhs, out_rhs)
end

@testset "model eval" begin
  task = DisentanglingVAETask()
  x_, y_ = FastAI.mocksample(task)
  x = FastAI.encodeinput(task, FastAI.Training(), x_)
  y = FastAI.encodetarget(task, FastAI.Training(), y_)
  xs = batch([x, x])
  ys = batch([y, y])

  model = VAE()
  loss = VAELoss{Float64}()
  ŷs = model(xs)
  @test !isnan(loss(ŷs, ys))
end

sample_latent(μ::AbstractArray{T}, logσ²::AbstractArray{T}) where {T} =
       μ .+ exp.(logσ²./2) .* randn!(similar(logσ²))

bernoulli_loss(x, logit_rec) = logitbinarycrossentropy(logit_rec, x;
                                                       agg=x->sum(x; dims=[1,2,3]))
kl_divergence(μ, logσ²; agg=mean) = sum(@. ((μ^2 + exp(logσ²) - 1 - logσ²) / 2); dims=1) |> agg
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
########################

#### Set up model #########
# image size is (64, 64)
# backbone_dim = 512
# backbone_dim = 768
# latent_dim = 64

resnet_backbone() = let backbone = Metalhead.ResNet(18; pretrain=false)
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
              Dense(128=>latent_dim),  # escape
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

# function FluxTraining.step!(learner, phase::VAETrainingPhase, batch)
#   (x_lhs, v_lhs, x_rhs, v_rhs, ks) = batch
#   params = Flux.params(learner.model.encoder, learner.model.bridge, learner.model.decoder)
#   FluxTraining.runstep(learner, phase, (; x_lhs=x_lhs, v_lhs=v_lhs, x_rhs=x_rhs, v_rhs=v_rhs, ks=ks)) do handle, state
#     gs = gradient(params) do

#       res = learner.model(state.x_lhs, state.v_lhs,
#                           state.x_rhs, state.v_rhs,
#                           state.ks_c)

#       handle(FluxTraining.LossBegin())
#       state.loss = (learner.lossfn(state.x_lhs, state.x̄_lhs, μ̂_lhs, logσ̂²_lhs;
#                                    warmup_factor=warmup_factor)
#                   + learner.lossfn(state.x_rhs, state.x̄_rhs, μ̂_rhs, logσ̂²_rhs;
#                                    warmup_factor=warmup_factor)
#                   # + 1f-1*warmup_factor*(cov_loss(state.z_lhs) + cov_loss(state.z_rhs))
#                   # + 1f-3*reg_l2(Flux.params(learner.model.decoder))  # we add some regularization here :)
#                   # + warmup_factor*directionality_loss(μ̂_lhs, μ̂_rhs)
#                   + mean((state.z_lhs .- state.v_lhs).^2)
#                   + mean((state.z_rhs .- state.v_rhs).^2)
#                     )

#       handle(FluxTraining.BackwardBegin())
#       return state.loss
#     end
#     if norm(gs) > 1f6
#       @warn "\n Large gradient with norm" norm(gs)
#     end
#     handle(FluxTraining.BackwardEnd())
#     Flux.Optimise.update!(learner.optimizer, params, gs)
#   end
# end

# mutable struct VAELoss{T}
#   reconstruction_loss
#   λ_kl::T
#   λ_l2_decoder::T
#   λ_cov::T
#   λ_directionality::T
#   λ_direct_supervision::T
# end
# using SimpleConfig, Configurations
# @option struct LossConfig end
# VAELoss(cfg::LossConfig) = VAELoss(
#     eval(Symbol(cfg.reconstruction_loss)), # string to symbol
#     cfg.λ_kl,
#     cfg.λ_l2_decoder,
#     cfg.λ_cov,
#     cfg.λ_directionality,
#     cfg.λ_direct_supervision)

# s = state
# function (loss::VAELoss)(s::FluxTraining.PropDict)
#   # ELBO w/ ELBO_tradeoff
#   # covariance loss
#   # decoder regularization
#   # directionality loss
#   # direct supervision
#   ( # ELBO
#     loss.reconstruction_loss(s.x_lhs, s.x̄_lhs) + loss.λ_kl * kl_divergence(s.μ̂_lhs, s.logσ̂²_lhs)
#   + loss.reconstruction_loss(s.x_rhs, s.x̄_rhs) + loss.λ_kl * kl_divergence(s.μ̂_rhs, s.logσ̂²_rhs)
#     # decoder regularization
#   + loss.λ_l2_decoder * reg_l2(Flux.params(learner.model.decoder))
#     # covariance regularization
#   + loss.λ_cov*(cov_loss(state.z_lhs) + cov_loss(state.z_rhs))
#     # directionality loss
#   + loss.λ_directionality*directionality_loss(μ̂_lhs, μ̂_rhs)
#     # direct supervision
#     + loss.λ_direct_supervision*(  mean((state.z_lhs .- state.v_lhs).^2)
#                                  + mean((state.z_rhs .- state.v_rhs).^2))
#    )
# end

function FluxTraining.step!(learner, phase::VAEValidationPhase, batch)
  (x_lhs, v_lhs, x_rhs, v_rhs, ks) = batch
  FluxTraining.runstep(learner, phase, (; x_lhs=x_lhs, v_lhs=v_lhs, x_rhs=x_rhs, v_rhs=v_rhs, ks=ks)) do handle, state
    @ignore_derivatives begin
      state.loss = ( learner.lossfn(state.x_lhs, learner.model(state.x_lhs)...)
                   + learner.lossfn(state.x_lhs, learner.model(state.x_lhs)...))
    end
  end
end

function FluxTraining.fit!(learner, nepochs::Int,
                           phases::Tuple{Pair{<:FluxTraining.AbstractTrainingPhase, <:Flux.DataLoader},
                                         Pair{<:FluxTraining.AbstractValidationPhase, <:Flux.DataLoader}})
    for _ in 1:nepochs
        for (phase, data) in phases
          epoch!(learner, phase, data)
        end
    end
end


############################################################
