import Configurations
import Configurations: @option
import Flux
import FluxTraining
import FluxTraining: HyperParameter
import ParameterSchedulers: Triangle, Sequence, Shifted
import FastAI: Scheduler, Learner
import StatsBase: mean
import Flux.Losses: logitbinarycrossentropy

reconstruction_loss = logitbinarycrossentropy  # "Bernoulli loss"
kl_divergence(μ, logσ²; agg=mean) = sum(@. ((μ^2 + exp(logσ²) - 1 - logσ²) / 2); dims=1) |> agg

@option mutable struct VAELoss{T}
  const reconstruction_loss::Function = reconstruction_loss
  λ_reconstruction::T     = 1.0
  λ_kl::T                 = 1.0
  λ_l2_decoder::T         = 0.0
  λ_covariance::T         = 0.0
  λ_directionality::T     = 0.0
  λ_direct_supervision::T = 0.0
  λ_escape_penalty::T     = 1000.0
end
Configurations.from_dict(::Type{VAELoss{T}}, ::Type{Function}, s) where T = eval(Symbol(s))

function (loss::VAELoss)(pred::Tuple, target::Tuple)
  pred   = (; lhs=(;x̄=pred[1],   μ=pred[2], logσ²=pred[3]),
              rhs=(;x̄=pred[4],   μ=pred[5], logσ²=pred[6]))
  target = (; lhs=(;x=target[1], v=target[2]),
              rhs=(;x=target[3], v=target[4]) )
                    
  ( # ELBO
    loss.λ_reconstruction * (  loss.reconstruction_loss(pred.lhs.x̄, target.lhs.x)
                             + loss.reconstruction_loss(pred.rhs.x̄, target.rhs.x) )
  + loss.λ_kl * ( kl_divergence(pred.lhs.μ, pred.lhs.logσ²)
                + kl_divergence(pred.rhs.μ, pred.rhs.logσ²) )
  )
end

# We subclass HyperParameter to interface with ParameterSchedulers.jl.
abstract type LossParam <: HyperParameter{Float64} end
abstract type Λ_reconstruction <: LossParam end
abstract type Λ_kl <: LossParam end
abstract type Λ_l2_decoder <: LossParam end
abstract type Λ_covariance <: LossParam end
abstract type Λ_directionality <: LossParam end
abstract type Λ_direct_supervision <: LossParam end
abstract type Λ_escape_penalty <: LossParam end
FluxTraining.stateaccess(::Type{<:LossParam}) = (lossfn = FluxTraining.Write(), )
FluxTraining.sethyperparameter!(learner, t::Type{<:LossParam}, val) = begin
  let sym = string(t) |> str->split(str, '.')[end] |> lowercase |> Symbol
    setfield!(learner.lossfn, sym, val)
  end
  return learner
end


@option struct LossSchedule
  λ_reconstruction_warmup::Integer     = 0
  λ_kl_warmup::Integer                 = 0
  λ_l2_decoder_warmup::Integer         = 0
  λ_covariance_warmup::Integer         = 0
  λ_directionality_warmup::Integer     = 0
  λ_direct_supervision_warmup::Integer = 0
  λ_escape_penalty_schedule_param::Integer = 0
end

@option struct LossConfig
  vae_loss::VAELoss{Float64} = VAELoss{Float64}()
  loss_schedule::LossSchedule = LossSchedule()
end

LinearWarmupSchedule(startlr, initlr, warmup_steps=-1) =
  Sequence(Triangle(λ0 = startlr, λ1 = initlr, period = 2 * warmup_steps) => warmup_steps,
           initlr => Inf)
@testset "loss scheduling" begin
  import Optimisers
  T = Float32
  xs = (rand(T, 32, 32, 3, 7),
        rand(T, 32, 32, 3, 7))
  ŷs = (rand(T, 32, 32, 3, 7), zeros(T, 6, 7), zeros(T, 6, 7),
         rand(T, 32, 32, 3, 7), zeros(T, 6, 7), zeros(T, 6, 7))
  ys = (rand(T, 32, 32, 3, 7), ones(T, 6, 7),
        rand(T, 32, 32, 3, 7), ones(T, 6, 7),
        zeros(T, 6, 7))


  local_reconstruction_loss(x_, x) = mean(x-x_)
  loss = VAELoss{Float64}(reconstruction_loss=local_reconstruction_loss)

  model = VAE()
  learner = Learner(model, loss; optimizer=Optimisers.Adam(),
                    callbacks=[Scheduler(Λ_reconstruction =>
                                                  LinearWarmupSchedule(0., 1., 10)),
                              ])

  initial_lossval = loss(ŷs, ys)
  @test !isnan(initial_lossval)

  losses = []
  for i in 1:5
    FluxTraining.step!(learner, VAETrainingPhase(), (xs, ys))
    push!(losses, loss(ŷs, ys))
  end
  @test all(losses/initial_lossval .≈ 0.0:0.1:0.4)
end
