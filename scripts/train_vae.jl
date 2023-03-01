# using Revise
using DisentanglingVAE
import DisentanglingVAE: make_experiment_path
import DisentanglingVAE: backbone, bridge, ResidualDecoder

import FastAI
import Flux
import StatsBase: sample, mean
import FastAI: ObsView, mapobs, taskdataloaders
import FastAI: TensorBoardBackend, LogMetrics, LogHyperParams
import Flux: cpu, gpu
import FluxTraining: fit!
import FastVision: ShowText, RGB
import ChainRulesCore: @ignore_derivatives
import MLUtils: _default_executor
import MLUtils.Transducers: ThreadedEx
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
                             DisentanglingVAE.LinearModelCallback(gpu),
                             LogMetrics(tb_backend)])
                             # LogHyperParams(tb_backend)])
                  # callbacks=[FastAI.ProgressPrinter(), ])

# test one input
# @ignore_derivatives model(FastAI.getbatch(learner)[1] |> DEVICE)
nepochs = 3000
fit!(learner, nepochs)
model_cpu = cpu(model);
@save joinpath(EXP_PATH, "model_ep_$nepochs.bson") model_cpu
#####################################################
