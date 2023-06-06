using SimpleConfig
using Configurations
include("config_utils.jl")

@option struct ModelConfig
  vae_loss::VAELoss{Float64} = VAELoss{Float64}()
  loss_schedule::LossSchedule = LossSchedule()
end

@option struct ExperimentConfig
  loss_cfg::LossConfig = LossConfig()
end

defaults = parse_defaults(ExperimentConfig())
cfg = define_configuration(ExperimentConfig, defaults)
println(cfg)
