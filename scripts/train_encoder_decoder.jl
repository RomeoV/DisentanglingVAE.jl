# This file is only to test whether the encoder and decoder are good enough by themselves.
using DisentanglingVAE
import DisentanglingVAE: make_experiment_path, make_data_sample
import DisentanglingVAE: backbone, bridge, ResidualDecoder, ResidualEncoder, ResidualEncoderWithHead, CSVLoggerBackend, log_to
import DisentanglingVAE: gaussian_nll

import FastAI
import FastAI: TensorBoardBackend, LogMetrics
import Flux
import StatsBase: sample, mean
import FastAI: load, datarecipes
import FastAI: mapobs, taskdataloaders
import FluxTraining: Checkpointer
import Flux: cpu, gpu
import FluxTraining: fit!
import MLUtils: _default_executor
import MLUtils.Transducers: ThreadedEx
import BSON: @save
import Distributions: Normal

_default_executor() = ThreadedEx()

function main_encoder()
    EXP_PATH = make_experiment_path()
    DRY = (isdefined(Main, :DRY) ? DRY : occursin("Romeo", read(`hostname`, String)))
    task = DisentanglingVAE.EncoderTask(32)

    ((cifar10_x, cifar10_y), blocks) = load(datarecipes()["cifar10"])
    airplane_img(i) = cifar10_x[(i % 15_000)+1]
    make_data_sample_(i::Int) = make_data_sample(Normal, i;
                                                x0_fn = i->0.1*airplane_img(i))[1:2]  # 15_000 airplanes
    n_datapoints=(DRY ? 2^10 : 2^14)
    data = mapobs(make_data_sample_, 1:n_datapoints)

    BATCHSIZE=(DRY ? 32 : 128)
    dl, dl_val = taskdataloaders(data, task, BATCHSIZE, pctgval=0.1;
                                buffer=false, partial=false,
                                );

    DEVICE = gpu
    dim_content, dim_style = 6, 0

    model = Flux.Chain(ResidualEncoder(128),  # <- output dim
                       bridge(128, dim_content+dim_style),)

    opt = Flux.Optimiser(Flux.ClipNorm(1.), Flux.Adam(3e-4))
    tb_backend = TensorBoardBackend(EXP_PATH)
    csv_backend = CSVLoggerBackend(EXP_PATH, 6)
    learner = FastAI.Learner(model, gaussian_nll;
                    optimizer=opt,
                    data=(dl, dl_val),
                    callbacks=[FastAI.ToGPU(),
                                FastAI.ProgressPrinter(),
                                LogMetrics((tb_backend, csv_backend)),
                                ExpDirPrinterCallback(EXP_PATH),
                                Checkpointer(EXP_PATH)])

    # test one input
    n_epochs=(DRY ? 3 : 1000)
    fit!(learner, n_epochs)
end
main_encoder()
