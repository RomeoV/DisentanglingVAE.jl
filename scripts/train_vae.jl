# using Revise
using DisentanglingVAE
import DisentanglingVAE: make_experiment_path, make_data_sample
import DisentanglingVAE: backbone, bridge, ResidualDecoder, ResidualEncoder, ResidualEncoderWithHead, CSVLoggerBackend, log_to

import FastAI
import Flux
import StatsBase: sample, mean
import FastAI: fitonecycle!, load, datarecipes
import FastAI: savetaskmodel, loadtaskmodel
import FastAI: ObsView, mapobs, taskdataloaders
import FastAI: TensorBoardBackend, LogMetrics, LogHyperParams
import FluxTraining
import FluxTraining: Checkpointer
import Flux: cpu, gpu
import FluxTraining: fit!, epoch!
import Wandb: WandbBackend
import FastVision: ShowText, RGB
import ChainRulesCore: @ignore_derivatives
import MLUtils: _default_executor
import MLUtils.Transducers: ThreadedEx
import BSON: @save, @load
import Distributions: Normal
@info "Starting train_vae.jl script"
# ThreadPoolEx gave me problems, see https://github.com/JuliaML/MLUtils.jl/issues/142
_default_executor() = ThreadedEx()

function main(; model_path=nothing)
    EXP_PATH = make_experiment_path()
    # DRY -> solve much smaller problem, usually for local machine
    DRY = (isdefined(Main, :DRY) ? DRY : occursin("Romeo", read(`hostname`, String)))
    DEVICE = gpu

    task, model = if isnothing(model_path)
        task = DisentanglingVAETask()

        dim_content, dim_style = 6, 0

        model = VAE(ResidualEncoder(128),  # <- output dim
                    bridge(128, dim_content+dim_style),
                    ResidualDecoder(dim_content+dim_style),
                    DEVICE);
        (task, model)
    else
        # loadtaskmodel(model_path)
        @load model_path model;
        DisentanglingVAETask(), model |> DEVICE
    end


    n_datapoints=(DRY ? 2^10 : 2^14)

    ((cifar10_x, cifar10_y), blocks) = load(datarecipes()["cifar10"])
    airplane_img(i) = cifar10_x[(i % 15_000)+1]
    make_data_sample_(i::Int) = make_data_sample(Normal(0, 0.5), i;
                                                 x0_fn = i->0.1*airplane_img(i))  # 15_000 airplanes
    data = mapobs(make_data_sample_, 1:n_datapoints)


    BATCHSIZE=(DRY ? 32 : 128)
    dl, dl_val = taskdataloaders(data, task, BATCHSIZE, pctgval=0.1;
                                buffer=false, partial=false,
                                );


    #### Try to run the training. #######################
    opt = Flux.Optimiser(Flux.ClipNorm(1.), Flux.Adam(3e-4))
    tb_backend = TensorBoardBackend(EXP_PATH)
    wandb_backend = WandbBackend(; project="DisentanglingVAE", entity="romeov")
    csv_backend = CSVLoggerBackend(EXP_PATH, 6)
    learner = FastAI.Learner(model, ELBO;
                    optimizer=opt,
                    data=(dl, dl_val),
                    callbacks=[FastAI.ToGPU(),
                               FastAI.ProgressPrinter(),
                               DisentanglingVAE.VisualizationCallback(task, gpu),
                               DisentanglingVAE.LinearModelCallback(gpu, ),
                               LogMetrics((tb_backend, csv_backend, wandb_backend)),
                               ExpDirPrinterCallback(EXP_PATH),
                               Checkpointer(EXP_PATH)])

    # test one input
    # @ignore_derivatives model(FastAI.getbatch(learner)[1] |> DEVICE)
    n_epochs=(DRY ? 200 : 1000)
    # fit!(learner, n_epochs, (VAETrainingPhase()=>dl, VAEValidationPhase()=>dl_val))
    for i in 1:n_epochs
      epoch!(learner, VAETrainingPhase(), dl)
      epoch!(learner, VAEValidationPhase(), dl_val)
      GC.gc()
    end
    # for _ in 1:4
    #     fitonecycle!(learner, 20, 3e-4;
    #                  phases=(VAETrainingPhase()   => dl,
    #                          VAEValidationPhase() => dl_val))
    # end
    close(wandb_backend)
    model_cpu = cpu(model);
    @save joinpath(EXP_PATH, "model_ep_$n_epochs.bson") model_cpu
    savetaskmodel(joinpath(EXP_PATH, "model_ep_$n_epochs.jld2"), task, learner.model)
    #####################################################
end
# main()

# fitonecycle!
