# using Revise
using DisentanglingVAE

using StatsBase: sample, mean
using FastAI: ObsView, mapobs, taskdataloaders
using FastAI.Flux: cpu, gpu
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

data = mapobs(DisentanglingVAE.make_data_sample, 1:2^13)

task = DisentanglingVAETask()

BATCHSIZE=128
dl, dl_val = taskdataloaders(data, task, BATCHSIZE, pctgval=0.1;
                             buffer=true, partial=false,
                            );

DEVICE = gpu
# DEVICE = cpu
model = VAE(backbone(), bridge(6), ResidualDecoder(6; sc=2), DEVICE);
params = Flux.params(model.bridge, model.decoder);

#### Try to run the training. #######################
opt = Flux.Optimiser(Flux.ClipNorm(1), Flux.Adam())
# opt = Flux.Optimiser(Adam())
learner = FastAI.Learner(model, ELBO;
                  optimizer=opt,
                  data=(dl, dl_val),
                  callbacks=[FastAI.ToGPU(),
                             FastAI.ProgressPrinter(),
                             DisentanglingVAE.VisualizationCallback(task, gpu)])
                  # callbacks=[FastAI.ProgressPrinter(), ])

# test one input
# @ignore_derivatives model(FastAI.getbatch(learner)[1] |> DEVICE)
FastAI.fitonecycle!(learner, 1, 1e-3;
                    div=10,
                    divfinal=1e2,
                    phases=(VAETrainingPhase() => dl,
                            VAEValidationPhase() => dl_val))
model_cpu = cpu(model);
@save "models/model2.bson" model_cpu
#####################################################

xs = FastAI.makebatch(task, data, rand(1:FastAI.numobs(data), 4)) |> DEVICE;
ys = model(xs; apply_sigmoid=true);
FastAI.showoutputbatch(BE, task, cpu.(xs), cpu.(ys))
