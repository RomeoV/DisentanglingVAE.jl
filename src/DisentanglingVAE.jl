module DisentanglingVAE
include("line_utils.jl")
include("losses.jl")
include("model.jl")
include("residual_models.jl")
include("task.jl")
include("callbacks.jl")
include("experiment_utils.jl")
include("fastai_patches.jl")
include("flux_patches.jl")
include("uncertainty_quantification.jl")

export DisentanglingVAETask, VAE, ELBO
export VAETrainingPhase, VAEValidationPhase
export VisualizationCallback, LinearModelCallback, ExpDirPrinterCallback
export ResidualBlock, ResidualEncoder, ResidualDecoder


import Flux
import FastVision
import FastAI: mapobs, taskdataloaders, Learner, fit!
# using PrecompileTools

# @setup_workload begin
#   DRY = true
#   @compile_workload begin
#       ((cifar10_x, cifar10_y), blocks) = load(datarecipes()["cifar10"])
#       make_data_sample_(i::Int) = make_data_sample(Normal, i; x0_fn = i->1//2*cifar10_x[(i % 15_000)+1])  # 15_000 airplanes
#       data = mapobs(make_data_sample_, 1:75)
#       task = DisentanglingVAETask()

#       dl, dl_val = taskdataloaders(data, task, 32, pctgval=0.1;
#                                    buffer=false, partial=false,
#                                   );
#       model = VAE(ResidualEncoder(128), bridge(128, 12),
#                   ResidualDecoder(12), gpu);
#       learner = Learner(model, ELBO;
#                         optimizer=Flux.Adam(3e-4),
#                         data=(dl, dl_val))
#       fit!(learner, 1)
#   end
# end

greet() = print("Hello World!")

end # module DisentanglingVAE
