# using Revise
using DisentanglingVAE
import DisentanglingVAE: make_experiment_path, make_data_sample
import DisentanglingVAE: backbone, bridge, ResidualDecoder, ResidualEncoder, ResidualEncoderWithHead, CSVLoggerBackend, log_to

import FastAI
import Flux
import StatsBase: sample, mean
import FastAI: fitonecycle!, load, datarecipes
import FastAI: ObsView, mapobs, taskdataloaders
import FastAI: TensorBoardBackend, LogMetrics, LogHyperParams
import FluxTraining: Checkpointer
import Flux: cpu, gpu
import FluxTraining: fit!
import FastVision: ShowText, RGB
import ChainRulesCore: @ignore_derivatives
import MLUtils: _default_executor
import MLUtils.Transducers: ThreadedEx
import BSON: @save
import Distributions: Normal
@info "Starting train_vae.jl script"
# ThreadPoolEx gave me problems, see https://github.com/JuliaML/MLUtils.jl/issues/142
_default_executor() = ThreadedEx()
BE = ShowText();  # sometimes get segfault by default
EXP_PATH = make_experiment_path()

n_datapoints=(occursin("Romeo", read(`hostname`, String)) ? 2^10 : 2^14)

((cifar10_x, cifar10_y), blocks) = load(datarecipes()["cifar10"])
make_data_sample_(i::Int) = make_data_sample(Normal, i; x0_fn = i->1//2*cifar10_x[(i % 15_000)+1])  # 15_000 airplanes
data = mapobs(make_data_sample_, 1:n_datapoints)

task = DisentanglingVAETask()

BATCHSIZE=(occursin("Romeo", read(`hostname`, String)) ? 32 : 128)
dl, dl_val = taskdataloaders(data, task, BATCHSIZE, pctgval=0.1;
                             buffer=false, partial=false,
                            );

DEVICE = gpu
# model = VAE(DisentanglingVAE.convnext_backbone(), bridge(6), ResidualDecoder(6; sc=1), DEVICE);
model = VAE(ResidualEncoder(128), bridge(128, 12), ResidualDecoder(12), DEVICE);

#### Try to run the training. #######################
opt = Flux.Optimiser(Flux.ClipNorm(1.), Flux.Adam(3e-4))
tb_backend = TensorBoardBackend(EXP_PATH)
csv_backend = CSVLoggerBackend(EXP_PATH, 6)
learner = FastAI.Learner(model, ELBO;
                  optimizer=opt,
                  data=(dl, dl_val),
                  callbacks=[FastAI.ToGPU(),
                             FastAI.ProgressPrinter(),
                             DisentanglingVAE.VisualizationCallback(task, gpu),
                             DisentanglingVAE.LinearModelCallback(gpu, ),
                             LogMetrics((tb_backend, csv_backend)),
                             ExpDirPrinterCallback(EXP_PATH),
                             Checkpointer(EXP_PATH)])
                             # LogHyperParams(tb_backend)])
                  # callbacks=[FastAI.ProgressPrinter(), ])

# test one input
# @ignore_derivatives model(FastAI.getbatch(learner)[1] |> DEVICE)
n_epochs=(occursin("Romeo", read(`hostname`, String)) ? 30 : 3000)
fit!(learner, n_epochs)
# fitonecycle!(learner, n_epochs;
#              div=100, divfinal=1, pct_start=30//n_epochs,
#              phases=(VAETrainingPhase() => dl,
#                      VAEValidationPhase() => dl_val))
model_cpu = cpu(model);
@save joinpath(EXP_PATH, "model_ep_$n_epochs.bson") model_cpu
#####################################################
