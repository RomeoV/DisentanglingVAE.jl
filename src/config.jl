using SimpleConfig
using Configurations

@option struct ModelConfig
  vae_loss::VAELoss{Float64} = VAELoss{Float64}()
  loss_schedule::LossSchedule = LossSchedule()
end

@option struct RuntimeConfig
  n_datapoints::Int = 2^14
  batch_size::Int = 2^9
end

@option struct ExperimentConfig
  loss_cfg::LossConfig = LossConfig()
  runtime_cfg::RuntimeConfig = RuntimeConfig()
end

## Some helpers for parsing default values.
parse_defaults(val::Number) = val
parse_defaults(vec::Vector) = parse_defaults.(vec)
parse_defaults(str::String) = str
parse_defaults(f::Function) = String(nameof(f))
function parse_defaults(cfg_struct) :: Dict
  fields = fieldnames(typeof(cfg_struct))
  Dict( String(f) => parse_defaults(getfield(cfg_struct, f))
        for f in fields )
end
