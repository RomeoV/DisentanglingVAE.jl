module DisentanglingVAE
# include("line_utils.jl")
using Reexport
include("VAEUtils/VAEUtils.jl")
@reexport using .VAEUtils
include("VAELosses/VAELosses.jl")
@reexport using .VAELosses
include("VAEData/VAEData.jl")
@reexport using .VAEData
include("VAETraining/VAETraining.jl")
@reexport using .VAETraining
include("VAECallbacks/VAECallbacks.jl")
@reexport using .VAECallbacks
include("VAEModels/VAEModels.jl")
@reexport using .VAEModels
import ReTest: @testset, @test
include("config.jl")
include("experiment_utils.jl")

export DisentanglingVAETask, VAE, VAELoss
export VAETrainingPhase, VAEValidationPhase
export VisualizationCallback, LinearModelCallback, ExpDirPrinterCallback
export LinearWarmupSchedule
export kl_divergence
export make_experiment_path

greet() = print("Hello World!")

# import PrecompileTools, FastAI, Optimisers
# PrecompileTools.@compile_workload begin
#     model = VAE()
#     task = DisentanglingVAETask()
#     data = FastAI.mapobs(make_data_sample, 1:16)
#     dl, dl_val = FastAI.taskdataloaders(data, task, 4, pctgval=0.25;
#                                  partial=false,
#                                  # buffer=false,
#                                  # parallel=false, # false for debugging
#                                  );
#     learner = FastAI.Learner(model, VAELoss{Float64}();
#                     optimizer=Optimisers.Adam(),
#                     data=(dl, dl_val))
#     FastAI.fit!(learner, 1)
# end

end # module DisentanglingVAE
