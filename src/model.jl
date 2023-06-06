import StatsBase: sample, mean
import ChainRulesCore: @ignore_derivatives
import Random: randn!
import Flux
import Flux: Dense, Parallel, Chain, LayerNorm, BatchNorm, Upsample, SamePad, leakyrelu, gradient, sigmoid, update!
import Flux.Losses: logitbinarycrossentropy
import FluxTraining
import FastAI
import FastAI: ToDevice, handle
import Optimisers
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

sample_latent(μ::AbstractArray{T, 2}, logσ²::AbstractArray{T, 2}) where {T} =
       μ .+ exp.(logσ²./2) .* randn!(similar(logσ²))

function (vae::VAE)(x::AbstractArray{T, 4}) where T
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
  task = DisentanglingVAETask()
  x_, y_ = FastAI.mocksample(task)
  # x = FastAI.encodeinput(task, FastAI.Training(), x_)
  y = FastAI.encodetarget(task, FastAI.Training(), y_)
  x = y[[1, 3]]
  xs = batch([x, x])
  ys = batch([y, y])

  model = VAE()
  loss = VAELoss{Float64}()
  ŷs = model(xs)
  lossval = loss(ŷs, ys)
  @test !isnan(lossval)
  @test lossval isa AbstractFloat
end

struct VAETrainingPhase <: FluxTraining.AbstractTrainingPhase end
struct VAEValidationPhase <: FluxTraining.AbstractValidationPhase end
function FluxTraining.on(::FluxTraining.StepBegin, ::Union{VAETrainingPhase, VAEValidationPhase},
                         cb::ToDevice, learner)
  # WARNING: this assumes that x and y[[1, 3]] are the same data!
  # we check this by only checking a few elements
  @assert all(learner.step.xs[1][1:5] .== learner.step.ys[1][1:5]) "$(learner.step.xs[1][1:5]) $(learner.step.ys[1][1:5])"
  @assert all(learner.step.xs[2][1:5] .== learner.step.ys[3][1:5]) "$(learner.step.xs[2][1:5]) $(learner.step.ys[3][1:5])"

  learner.step.ys = cb.movedatafn.(learner.step.ys)
  # we only want to move the data to gpu once.
  learner.step.xs = (copy(learner.step.ys[1]), copy(learner.step.ys[3]))
end

function FluxTraining.step!(learner, phase::VAETrainingPhase, batch)
    xs, ys = batch
    FluxTraining.runstep(learner, phase, (; xs=xs, ys=ys)) do handle, state
        state.grads = FluxTraining._gradient(learner.optimizer, learner.model, learner.params) do model
            x_lhs, x_rhs = state.xs
            _x_lhs, v_lhs, _x_rhs, v_rhs, ks_c = state.ys
            μ_lhs, logσ²_lhs, escape_lhs = model.encoder(x_lhs)
            μ_rhs, logσ²_rhs, escape_rhs = model.encoder(x_rhs)

            # averaging mask with 1s for all style variables (which we always average)
            ks_cs = @ignore_derivatives let sz = (size(μ_lhs, 1) - size(ks_c, 1), size(ks_c, 2))
                ks_style = 1 .+ 0 .* similar(ks_c, sz)
                vcat(ks_c, ks_style)
            end

            μ̂_lhs     = ks_cs.*(μ_lhs+μ_rhs)./2         + (1 .- ks_cs).*(μ_lhs)
            μ̂_rhs     = ks_cs.*(μ_lhs+μ_rhs)./2         + (1 .- ks_cs).*(μ_rhs)
            logσ̂²_lhs = ks_cs.*(logσ²_lhs+logσ²_rhs)./2 + (1 .- ks_cs).*(logσ²_lhs)
            logσ̂²_rhs = ks_cs.*(logσ²_lhs+logσ²_rhs)./2 + (1 .- ks_cs).*(logσ²_rhs)

            z_lhs     = sample_latent(μ̂_lhs, logσ̂²_lhs)  # + escape_lhs
            z_rhs     = sample_latent(μ̂_rhs, logσ̂²_rhs)  # + escape_rhs
            x̄_lhs     = model.decoder(z_lhs)
            x̄_rhs     = model.decoder(z_rhs)

            state.ŷs = (x̄_lhs, μ̂_lhs, logσ̂²_lhs, x̄_rhs, μ̂_rhs, logσ̂²_rhs)
            handle(FluxTraining.LossBegin())

            state.loss = let escape_layer = model.encoder.layers[2].layers[end].layers[end],
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
        learner.params, learner.model = FluxTraining._update!(
            learner.optimizer, learner.params, learner.model, state.grads)
    end
end
FluxTraining.step!(learner, phase::VAEValidationPhase, batch) =
    FluxTraining.step!(learner, FluxTraining.ValidationPhase(), batch)

@testset "model step" for device in [Flux.cpu, Flux.gpu]
  model = VAE()
  learner = Learner(model, VAELoss{Float64}();
                    optimizer=Optimisers.Adam(),
                    callbacks=[FluxTraining.ToDevice(device, device)])

  task = DisentanglingVAETask()
  x_, y_ = FastAI.mocksample(task)
  # x = FastAI.encodeinput(task, FastAI.Training(), x_)
  y = FastAI.encodetarget(task, FastAI.Training(), y_)
  x = y[[1, 3]]
  xs = batch([x, x])
  ys = batch([y, y])

  FluxTraining.step!(learner, VAETrainingPhase(), (xs, ys))
  @test true
end

"We override fit!, which is necessary to use the VAE{Training,Validation}Phase
 instead of the normal {Training,Validation}Phase, so that we can use our own
 step! and data movement functions."
function FluxTraining.fit!(learner, nepochs::Int,
                           phases::Tuple{Pair{<:FluxTraining.AbstractTrainingPhase, <:Flux.DataLoader},
                                         Pair{<:FluxTraining.AbstractValidationPhase, <:Flux.DataLoader}})
    for _ in 1:nepochs
        for (phase, data) in phases
          FluxTraining.epoch!(learner, phase, data)
        end
    end
end
