import StatsBase: sample, mean
import ChainRulesCore: @ignore_derivatives
import Random: randn!
import CUDA
import Flux
import Flux: Dense, Parallel, Chain, LayerNorm, BatchNorm, Upsample, SamePad, leakyrelu, gradient, sigmoid, update!
import Flux.Losses: logitbinarycrossentropy
import FluxTraining
import FastAI
import FastAI: ToDevice, handle
import Metalhead
import LinearAlgebra: norm
import Match: @match
import MLUtils: batch

struct VAE{E, D}
  encoder::E
  decoder::D
end
Flux.@functor VAE
VAE() = VAE(Chain(ResidualEncoder(128), DisentanglingVAE.bridge(128, 6)),
            ResidualDecoder(6))

const AbstractImageTensor = AbstractArray{T, 4} where T
function (vae::VAE)(x::AbstractImageTensor{T}) where T
  μ, logσ², escape = vae.encoder(x)
  z = sample_latent(μ, logσ²) + escape
  x̄ = vae.decoder(z)
  return x̄, μ, logσ²
end

function (vae::VAE)((x_lhs, v_lhs, x_rhs, v_rhs, ks_c)::Tuple{<:AbstractImageTensor{T},
                                                              <:AbstractMatrix{T},
                                                              <:AbstractImageTensor{T},
                                                              <:AbstractMatrix{T},
                                                              <:AbstractMatrix{T}}) where T
  return (vae(x_lhs)..., vae(x_rhs)...)
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
  lossval = loss(ŷs, ys)
  @test !isnan(lossval)
  @test lossval isa AbstractFloat
end

bernoulli_loss(x, logit_rec) = logitbinarycrossentropy(logit_rec, x;
                                                       agg=x->sum(x; dims=[1,2,3]))
kl_divergence(μ, logσ²; agg=mean) = sum(@. ((μ^2 + exp(logσ²) - 1 - logσ²) / 2); dims=1) |> agg

struct VAETrainingPhase <: FluxTraining.AbstractTrainingPhase end
struct VAEValidationPhase <: FluxTraining.AbstractValidationPhase end

# function FluxTraining.on(
#     ::FluxTraining.StepBegin,
#     ::Union{VAETrainingPhase, VAEValidationPhase},
#     cb::ToDevice,
#     learner,
#   )
#   learner.step.x_lhs = cb.movedatafn(learner.step.x_lhs)
#   learner.step.v_lhs = cb.movedatafn(learner.step.v_lhs)
#   learner.step.x_rhs = cb.movedatafn(learner.step.x_rhs)
#   learner.step.v_rhs = cb.movedatafn(learner.step.v_rhs)
#   learner.step.ks    = begin T=eltype(learner.step.x_lhs);
#     cb.movedatafn(convert.(T, learner.step.ks))
#   end
# end

sample_latent(μ::AbstractArray{T}, logσ²::AbstractArray{T}) where {T} =
       μ .+ exp.(logσ²./2) .* randn!(similar(logσ²))

function FluxTraining.step!(learner, phase::VAETrainingPhase, batch)
    xs, ys = batch
    FluxTraining.runstep(learner, phase, (; xs=xs, ys=ys)) do handle, state
        state.grads = gradient(learner.params) do
            x_lhs, v_lhs, x_rhs, v_rhs, ks_c = xs
            μ_lhs, logσ²_lhs, escape_lhs = learner.model.encoder(x_lhs)
            μ_rhs, logσ²_rhs, escape_rhs = learner.model.encoder(x_rhs)

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
            x̄_lhs     = learner.model.decoder(z_lhs)
            x̄_rhs     = learner.model.decoder(z_rhs)

            state.ŷs = (x̄_lhs, μ̂_lhs, logσ̂²_lhs, x̄_rhs, μ̂_rhs, logσ̂²_rhs)
            handle(FluxTraining.LossBegin())

            state.loss = let escape_layer = learner.model.encoder.layers[2].layers[end].layers[end],
                             loss = learner.lossfn
              ( # ELBO 
                loss(state.ŷs, state.ys)
                # escape penalty
              + loss.λ_escape_penalty * reg_l1(Flux.params(escape_layer))
                # decoder regularization
              + loss.λ_l2_decoder * reg_l2(Flux.params(learner.model.decoder))
                # covariance regularization
              + loss.λ_covariance * (cov_loss(z_lhs) + cov_loss(z_rhs))
                # directionality loss
              + loss.λ_directionality * directionality_loss(μ̂_lhs, μ̂_rhs)
                # direct supervision
              + loss.λ_direct_supervision * (  Flux.mse(z_lhs, v_lhs)
                                             + Flux.mse(z_rhs, v_rhs) )
              )
            end
            handle(FluxTraining.BackwardBegin())
            return state.loss
        end
        handle(FluxTraining.BackwardEnd())
        update!(learner.optimizer, learner.params, state.grads)
    end
end

@testset "model step" begin
  model = VAE()
  learner = Learner(model, VAELoss{Float64}(); optimizer=Flux.Adam())

  task = DisentanglingVAETask()
  x_, y_ = FastAI.mocksample(task)
  x = FastAI.encodeinput(task, FastAI.Training(), x_)
  y = FastAI.encodetarget(task, FastAI.Training(), y_)
  xs = batch([x, x])
  ys = batch([y, y])

  FluxTraining.step!(learner, VAETrainingPhase(), (xs, ys))
  @test true
end

" Necessary to use the VAE{Training,Validation}Phase instead of the normal {Training,Validation}Phase. "
function FluxTraining.fit!(learner, nepochs::Int,
                           phases::Tuple{Pair{<:FluxTraining.AbstractTrainingPhase, <:Flux.DataLoader},
                                         Pair{<:FluxTraining.AbstractValidationPhase, <:Flux.DataLoader}})
    for _ in 1:nepochs
        for (phase, data) in phases
          epoch!(learner, phase, data)
        end
    end
end
