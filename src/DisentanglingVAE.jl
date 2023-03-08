module DisentanglingVAE
include("line_utils.jl")
include("model.jl")
include("residual_models.jl")
include("task.jl")
include("callbacks.jl")
include("experiment_utils.jl")
include("fastai_patches.jl")
include("flux_patches.jl")

export DisentanglingVAETask, VAE, ELBO
export VAETrainingPhase, VAEValidationPhase
export VisualizationCallback, LinearModelCallback
export ResidualBlock, ResidualEncoder, ResidualDecoder

greet() = print("Hello World!")

end # module DisentanglingVAE
