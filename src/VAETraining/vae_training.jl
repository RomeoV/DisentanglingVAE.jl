import Flux
import FluxTraining
import ChainRulesCore: @ignore_derivatives
import ..VAEUtils: sample_latent

struct VAETrainingPhase <: FluxTraining.AbstractTrainingPhase end
struct VAEValidationPhase <: FluxTraining.AbstractValidationPhase end

function FluxTraining.on(::FluxTraining.StepBegin, ::Union{VAETrainingPhase, VAEValidationPhase},
                         cb::FluxTraining.ToDevice, learner)
  # WARNING: this assumes that x and y[[1, 3]] are the same data!
  # we check this by only checking a few elements
  # @assert all(learner.step.xs[1][1:5] .== learner.step.ys[1][1:5]) "$(learner.step.xs[1][1:5]) $(learner.step.ys[1][1:5])"
  # @assert all(learner.step.xs[2][1:5] .== learner.step.ys[3][1:5]) "$(learner.step.xs[2][1:5]) $(learner.step.ys[3][1:5])"

  learner.step.xs = cb.movedatafn.(learner.step.xs)
  learner.step.ys = cb.movedatafn.(learner.step.ys)
  # we only want to move the data to gpu once.
  # learner.step.xs = (copy(learner.step.ys[1]), copy(learner.step.ys[3]))
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
            state.loss = ( # ELBO
                learner.lossfn(state.ŷs, state.ys)
              + learner.lossfn(state.ŷs, state.ys, learner.model, phase)  # this is the regularization part only applied during training
            )
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
