module DisentanglingVAE
using FastAI
using FastAI.Flux
using Metalhead
include("line_utils.jl")
include("model.jl")
include("residual_models.jl")
include("task.jl")
include("callbacks.jl")
include("experiment_utils.jl")

export DisentanglingVAETask, VAE, ELBO
export VAETrainingPhase, VAEValidationPhase

greet() = print("Hello World!")

end # module DisentanglingVAE
