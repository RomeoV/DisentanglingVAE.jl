using Revise
using DisentanglingVAE

using StatsBase: sample, mean
using FastAI: ObsView, mapobs, taskdataloaders
using FastAI.Flux: cpu, gpu
import FastAI
import FastAI.Flux
import FastVision: ShowText, RGB
import FixedPointNumbers: N0f8
using Random: seed!
using DisentanglingVAE: backbone, bridge, ResidualDecoder
import ChainRulesCore: @ignore_derivatives
import FastAI.Flux.MLUtils: _default_executor
import FastAI.MLUtils.Transducers: ThreadedEx
# ThreadPoolEx gave me problems, see https://github.com/JuliaML/MLUtils.jl/issues/142
_default_executor() = ThreadedEx()
BE = ShowText();  # sometimes get segfault by default

function make_data_sample(i)
  seed!(i)
  ks = rand(Bool, 6)
  std = 1/sqrt(2)
  v_lhs = randn(Float32, 6)*std
  v_rhs = randn(Float32, 6)*std
  v_rhs[ks] .= v_lhs[ks]

  img_lhs = zeros(RGB{N0f8}, 64, 64)
  DisentanglingVAE.draw!(img_lhs, v_lhs[1:2]..., RGB{N0f8}(1.,0,0))
  DisentanglingVAE.draw!(img_lhs, v_lhs[3:4]..., RGB{N0f8}(0,1.,0))
  DisentanglingVAE.draw!(img_lhs, v_lhs[5:6]..., RGB{N0f8}(0,0,1.))

  img_rhs = zeros(RGB{N0f8}, 64, 64)
  DisentanglingVAE.draw!(img_rhs, v_rhs[1:2]..., RGB{N0f8}(1.,0,0))
  DisentanglingVAE.draw!(img_rhs, v_rhs[3:4]..., RGB{N0f8}(0,1.,0))
  DisentanglingVAE.draw!(img_rhs, v_rhs[5:6]..., RGB{N0f8}(0,0,1.))

  (img_lhs, v_lhs, img_rhs, v_rhs, ks)
end
data = mapobs(make_data_sample, 1:2^13)

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
FastAI.fitonecycle!(learner, 50, 1e-4;
                    phases=(VAETrainingPhase() => dl,
                            VAEValidationPhase() => dl_val))
#####################################################

xs = FastAI.makebatch(task, data, rand(1:FastAI.numobs(data), 4)) |> DEVICE;
ys = model(xs; apply_sigmoid=true);
FastAI.showoutputbatch(BE, task, cpu.(xs), cpu.(ys))
