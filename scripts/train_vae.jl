# using Revise
using DisentanglingVAE
import DisentanglingVAE: make_experiment_path

using StatsBase: sample, mean
using FastAI: ObsView, mapobs, taskdataloaders
using FastAI: TensorBoardBackend, LogMetrics, LogHyperParams
using FastAI.Flux: cpu, gpu
import FastAI.FluxTraining: fit!
import FastAI
import FastAI.Flux
import FastVision: ShowText, RGB
using DisentanglingVAE: backbone, bridge, ResidualDecoder
import ChainRulesCore: @ignore_derivatives
import FastAI.Flux.MLUtils: _default_executor
import FastAI.MLUtils.Transducers: ThreadedEx
import BSON: @save
# ThreadPoolEx gave me problems, see https://github.com/JuliaML/MLUtils.jl/issues/142
_default_executor() = ThreadedEx()
BE = ShowText();  # sometimes get segfault by default
EXP_PATH = make_experiment_path()

data = mapobs(DisentanglingVAE.make_data_sample, 1:2^13)

task = DisentanglingVAETask()

BATCHSIZE=128
dl, dl_val = taskdataloaders(data, task, BATCHSIZE, pctgval=0.1;
                             buffer=false, partial=false,
                            );

DEVICE = gpu
# DEVICE = cpu
model = VAE(backbone(), bridge(6), ResidualDecoder(6; sc=2), DEVICE);
params = Flux.params(model.bridge, model.decoder);

#### Try to run the training. #######################
opt = Flux.Optimiser(Flux.ClipNorm(1), Flux.Adam(3e-4))
# opt = Flux.Optimiser(Adam())
tb_backend = TensorBoardBackend(EXP_PATH)
learner = FastAI.Learner(model, ELBO;
                  optimizer=opt,
                  data=(dl, dl_val),
                  callbacks=[FastAI.ToGPU(),
                             FastAI.ProgressPrinter(),
                             DisentanglingVAE.VisualizationCallback(task, gpu),
                             LogMetrics(tb_backend),
                             LogHyperParams(backend)])
                  # callbacks=[FastAI.ProgressPrinter(), ])

# test one input
# @ignore_derivatives model(FastAI.getbatch(learner)[1] |> DEVICE)
fit!(learner, 1000)
model_cpu = cpu(model);
@save joinpath(EXP_PATH, "model_ep_$epoch.bson") model_cpu
#####################################################
