using InlineTest
using Configurations
import FluxTraining
import FluxTraining: HyperParameter

reconstruction_loss(x_, x) = sum(x-x_)

@option mutable struct VAELoss{T}
  const reconstruction_loss::Function = reconstruction_loss
  λ_reconstruction::T     = 1.0
  λ_kl::T                 = 1.0
  λ_l2_decoder::T         = 1.0
  λ_covariance::T         = 1.0
  λ_directionality::T     = 1.0
  λ_direct_supervision::T = 1.0
end
Configurations.from_dict(::Type{VAELoss{T}}, ::Type{Function}, s) where T = eval(Symbol(s))

# function (loss::VAELoss)(pred::VAEResultDouble, target::Tuple)
#   target = (; lhs=(;x=target[1], v=target[2]),
#               rhs=(;x=target[3], v=target[4]) )
                    
#   ( # ELBO
#     loss.λ_reconstruction * (  loss.reconstruction_loss(pred.lhs.x̄, target.lhs.x)
#                              + loss.reconstruction_loss(pred.rhs.x̄, target.rhs.x) )
#   + loss.λ_kl * ( kl_divergence(pred.lhs.μ̂, pred.lhs.logσ̂²)
#                 + kl_divergence(pred.rhs.μ̂, pred.rhs.logσ̂²) )
#     # decoder regularization
#   # + loss.λ_l2_decoder * reg_l2(Flux.params(vae.decoder))
#     # covariance regularization
#   + loss.λ_covariance * (cov_loss(pred.lhs.z) + cov_loss(pred.rhs.z))
#     # directionality loss
#   + loss.λ_directionality * directionality_loss(pred.lhs.μ̂, pred.rhs.μ̂)
#     # direct supervision
#   + loss.λ_direct_supervision * (  Flux.mse(pred.lhs.z, target.lhs.v)
#                                  + Flux.mse(pred.rhs.z, target.rhs.v) )
#     # escape penalty
#   # + loss.λ_escape_penalty * reg_l1(Flux.params(vae.encoder.layers[2].layers[end].layers[end]))
#   )
# end
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

OutType = Tuple{<:AbstractArray{<:Number, 4},
                <:AbstractArray{<:Number, 2},
                <:AbstractArray{<:Number, 4},
                <:AbstractArray{<:Number, 2}}

function (loss::VAELoss)(ŷ::OutType, y::OutType)
  x̄_lhs, z_lhs, x̄_rhs, z_rhs = ŷ
  x_lhs, v_lhs, x_rhs, v_rhs = ŷ
  ( loss.reconstruction_loss(x̄_lhs, x_lhs)
  + loss.reconstruction_loss(x̄_rhs, x_rhs)
  + kl_divergence(s.μ̂_rhs, s.logσ̂²_rhs)
  + kl_divergence(s.μ̂_lhs, s.logσ̂²_lhs) )
end

# We need this for FluxTraining.fit!
ELBO((x, x̄, μ, logσ²)::Tuple; warmup_factor::Rational=1//1) = ELBO(x, x̄, μ, logσ²;
                                                                   warmup_factor=warmup_factor)
ELBO((x̄, μ, logσ²)::Tuple, x; warmup_factor::Rational=1//1) = ELBO(x, x̄, μ, logσ²;
                                                                   warmup_factor=warmup_factor)




# We subclass HyperParameter to interface with ParameterSchedulers.jl.
abstract type LossParam <: HyperParameter{Float64} end
abstract type Λ_reconstruction <: LossParam end
abstract type Λ_kl <: LossParam end
abstract type Λ_l2_decoder <: LossParam end
abstract type Λ_covariance <: LossParam end
abstract type Λ_directionality <: LossParam end
abstract type Λ_direct_supervision <: LossParam end

@option struct LossSchedule
  λ_reconstruction_warmup::Integer     = 0
  λ_kl_warmup::Integer                 = 0
  λ_l2_decoder_warmup::Integer         = 0
  λ_covariance_warmup::Integer         = 0
  λ_directionality_warmup::Integer     = 0
  λ_direct_supervision_warmup::Integer = 0
end

@option struct LossConfig
  vae_loss::VAELoss{Float64} = VAELoss{Float64}()
  loss_schedule::LossSchedule = LossSchedule()
end

@testset "loss" begin
  @test 1==1
end
