module DisentanglingVAE
# include("line_utils.jl")
include("flux_patches.jl")
include("fastai_patches.jl")
include("data.jl")
include("losses.jl")
include("loss.jl")
include("config.jl")
include("layers.jl")
include("residual_models.jl")
include("model.jl")
include("task.jl")
include("callbacks.jl")
include("experiment_utils.jl")
include("uncertainty_quantification.jl")

export DisentanglingVAETask, VAE, ELBO
export VAETrainingPhase, VAEValidationPhase
export VisualizationCallback, LinearModelCallback, ExpDirPrinterCallback
export ResidualBlock, ResidualEncoder, ResidualDecoder
export LinearWarmupSchedule
import CUDA


# import Flux
# import FastVision
# import FastAI: mapobs, taskdataloaders, Learner, fit!
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

import PrecompileTools
@PrecompileTools.compile_workload begin
  import DisentanglingVAE.LineData: make_data_sample
  _data = make_data_sample(1)

  task = DisentanglingVAETask()
  x_, y_ = FastAI.mocksample(task)
  x = FastAI.encodeinput(task, FastAI.Training(), x_)
  y = FastAI.encodetarget(task, FastAI.Training(), y_)
  xs = batch([x, x])
  ys = batch([y, y])

  # model = VAE()
  # loss = VAELoss{Float64}()
  # ŷs = model(xs)
  # lossval = loss(ŷs, ys)
  # learner = Learner(model, loss; optimizer=Flux.Adam(),
  #                   callbacks=[
  #                     Scheduler(Λ_reconstruction =>
  #                               LinearWarmupSchedule(0., 1., 10)),
  #                             ])
  # for i in 1:5
  #   FluxTraining.step!(learner, VAETrainingPhase(), (xs, ys))
  # end
end

end # module DisentanglingVAE
