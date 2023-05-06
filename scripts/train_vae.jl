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

function main()
    EXP_PATH = make_experiment_path()
    # DRY -> solve much smaller problem, usually for local machine
    DRY = (isdefined(Main, :DRY) ? DRY : occursin("Romeo", read(`hostname`, String)))

    n_datapoints=(DRY ? 2^10 : 2^14)

    ((cifar10_x, cifar10_y), blocks) = load(datarecipes()["cifar10"])
    airplane_img(i) = cifar10_x[(i % 15_000)+1]
    make_data_sample_(i::Int) = make_data_sample(Normal, i;
                                                 x0_fn = i->0.1*airplane_img(i))  # 15_000 airplanes
    data = mapobs(make_data_sample_, 1:n_datapoints)

    task = DisentanglingVAETask()

    BATCHSIZE=(DRY ? 32 : 128)
    dl, dl_val = taskdataloaders(data, task, BATCHSIZE, pctgval=0.1;
                                buffer=false, partial=false,
                                );

    DEVICE = gpu
    dim_content, dim_style = 6, 2

    model = VAE(ResidualEncoder(128),  # <- output dim
                bridge(128, dim_content+dim_style),
                ResidualDecoder(dim_content+dim_style),
                DEVICE);

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

    # test one input
    # @ignore_derivatives model(FastAI.getbatch(learner)[1] |> DEVICE)
    n_epochs=(DRY ? 3 : 1000)
    fit!(learner, n_epochs;
         phases=(VAETrainingPhase()=>dl, VAEValidationPhase()=>dl_val))
    # fitonecycle!(learner, n_epochs;
    #              div=100, divfinal=1, pct_start=30//n_epochs,
    #              phases=(VAETrainingPhase() => dl,
    #                      VAEValidationPhase() => dl_val))
    model_cpu = cpu(model);
    @save joinpath(EXP_PATH, "model_ep_$n_epochs.bson") model_cpu
    #####################################################
end
main()

# fitonecycle!
