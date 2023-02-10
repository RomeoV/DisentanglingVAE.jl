import ChainRulesCore: @ignore_derivatives
import Random: randn!
import FastAI.Flux.Losses: logitbinarycrossentropy

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

function (vae::VAE)(x)
  intermediate = vae.encoder(x)
  μ, logσ² = vae.bridge(intermediate)
  z = sample_latent(μ, logσ²)
  x̄ = vae.decoder(z)
  return x̄, (; μ, logσ²)
end

sample_latent(μ::AbstractArray{T}, logσ²::AbstractArray{T}) where {T} =
       μ .+ exp.(logσ²./2) .* randn!(similar(logσ²))

bernoulli_loss(x, x_rec) = logitbinarycrossentropy(x_rec, x;
                                                   agg=x->sum(x; dims=[1,2,3]))
function ELBO(x, x̄, μ, logσ²)
  # reconstruction_error = mean(sum(@. ((x̄ - x)^2); dims=(1,2,3)))
  reconstruction_error = bernoulli_loss(x, x̄)
  kl_divergence = sum(@. ((μ^2 + exp(logσ²) - 1 - logσ²) / 2); dims=1)
  return mean(reconstruction_error) + mean(kl_divergence)
end
########################

#### Set up model #########
# image size is (64, 64)
backbone_dim = 512
latent_dim = 64

backbone() = begin backbone = Metalhead.ResNet(18; pretrain=true)
   Chain(backbone.layers[1], Chain(backbone.layers[2].layers[1:2]..., identity))
 end

bridge() = 
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
  learner.step.x_rhs = cb.movedatafn(learner.step.x_rhs)
end

function FluxTraining.step!(learner, phase::VAETrainingPhase, batch)
  FluxTraining.runstep(learner, phase, (; x=batch[1], x_lhs=batch[1], v_lhs=batch[2], x_rhs=batch[3], v_rhs=batch[4], ks = batch[5], )) do handle, state
    gs = gradient(params) do
      intermediate_lhs   = @ignore_derivatives learner.model.encoder(state.x_lhs)
      intermediate_rhs   = @ignore_derivatives learner.model.encoder(state.x_rhs)
      μ_lhs, logσ²_lhs   = learner.model.bridge(intermediate_lhs)
      μ_rhs, logσ²_rhs   = learner.model.bridge(intermediate_rhs)
      μ, logσ²       = μ_rhs, logσ²_rhs
      state.z        = sample_latent(μ, logσ²)
      state.x̄        = learner.model.decoder(state.z)
      state.y = state.x
      state.ŷ = state.x̄

      handle(FluxTraining.LossBegin())
      state.loss = learner.lossfn(state.x_lhs, state.x̄, μ, logσ²)

      handle(FluxTraining.BackwardBegin())
      return state.loss
    end
    handle(FluxTraining.BackwardEnd())
    Flux.Optimise.update!(learner.optimizer, params, gs)
  end
end

function FluxTraining.step!(learner, phase::VAEValidationPhase, batch)
  FluxTraining.runstep(learner, phase, (; x=batch[1], x_lhs=batch[1], v_lhs=batch[2], x_rhs=batch[3], v_rhs=batch[4], ks = batch[5], )) do handle, state
    @ignore_derivatives begin
      intermediate   = learner.model.encoder(state.x_lhs)
      μ, logσ²       = learner.model.bridge(intermediate)
      state.z        = sample_latent(μ, logσ²)
      state.x̄        = learner.model.decoder(state.z)
      state.loss = learner.lossfn(state.x_lhs, state.x̄, μ, logσ²)
    end
  end
end
############################################################
