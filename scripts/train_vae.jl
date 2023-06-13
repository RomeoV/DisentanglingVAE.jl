# using Revise
using DisentanglingVAE
import DisentanglingVAE: make_experiment_path
import DisentanglingVAE: VAELoss
import DisentanglingVAE: VisualizationCallback, LinearModelCallback, CSVLoggerBackend
import DisentanglingVAE: ExperimentConfig, parse_defaults
import DisentanglingVAE.VAELosses: Λ_kl, Λ_escape_penalty, LinearWarmupSchedule
import DisentanglingVAE.VAEData: make_data_sample
import DisentanglingVAE.VAECallbacks: ExpDirPrinterCallback

import CUDA
import FastAI, FastVision
import Flux
import Flux.Losses: logitbinarycrossentropy
import Optimisers
import FastAI: fitonecycle!, load, datarecipes
import FastAI: savetaskmodel, loadtaskmodel
import FastAI: ObsView, mapobs, taskdataloaders
import FastAI: TensorBoardBackend, LogMetrics, LogHyperParams
import FastAI: GarbageCollect
import FluxTraining
import FluxTraining: Checkpointer
import Flux: cpu, gpu, Chain
import FluxTraining: fit!, epoch!
import Wandb: WandbBackend
import FastVision: ShowText, RGB
import ChainRulesCore: @ignore_derivatives
import MLUtils
import BSON: @save, @load
import Distributions: Normal
import SimpleConfig: define_configuration

import MLUtils.Transducers: ThreadedEx
# ThreadPoolEx gave me problems, see https://github.com/JuliaML/MLUtils.jl/issues/142
MLUtils._default_executor() = MLUtils.Transducers.ThreadedEx()

CUDA.allowscalar(false)

function main(; model_path=nothing)
    @info "Starting VAE training."
    experiment_config_defaults = parse_defaults(ExperimentConfig())
    cfg = define_configuration(ExperimentConfig, experiment_config_defaults)
    rt_cfg = cfg.runtime_cfg
    # cfg = (; n_datapoints=2^14,
    #          batch_size=2^9,    )

    EXP_PATH = make_experiment_path()
    # DRY -> solve much smaller problem, usually for local machine
    DRY = (isdefined(Main, :DRY) ? Main.DRY : occursin("Romeo", read(`hostname`, String)))
    DEVICE = gpu

    task = DisentanglingVAETask()

    model = VAE()

    loss = VAELoss()
    loss_scheduler = FastAI.Scheduler(
        Λ_kl  => LinearWarmupSchedule(0., 1., 100),
        Λ_escape_penalty => LinearWarmupSchedule(0., 100., 10_000),
    )


    # Note: choose DRY n_datapoints and batch_size such that
    # the number of steps per epoch stays the same :).
    # 2^14 -> 2^11
    # 2^7 -> 2^4
    n_datapoints=(DRY ? rt_cfg.n_datapoints÷(2^3) : rt_cfg.n_datapoints)
    data = mapobs(make_data_sample, 1:n_datapoints)

    batch_size=(DRY ? rt_cfg.batch_size÷(2^3) : rt_cfg.batch_size)
    dl, dl_val = taskdataloaders(data, task, batch_size, pctgval=0.1;
                                 buffer=true, 
                                 partial=false,
                                 parallel=true, # false for debugging
                                 );

    opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(1f0),
                                    Optimisers.Adam(3e-4))
    logging_backends = [
        TensorBoardBackend(EXP_PATH),
        # wandb_backend = WandbBackend(; project="DisentanglingVAE", entity="romeov"),
        CSVLoggerBackend(EXP_PATH, 6),
    ]
    learner = FastAI.Learner(model, loss;
                    optimizer=opt,
                    data=(dl, dl_val),
                    callbacks=[FastAI.ToGPU(),
                               FastAI.ProgressPrinter(),
                               VisualizationCallback(task=task, device=gpu),
                               # LinearModelCallback(gpu, ),
                               # LogMetrics(logging_backends...),
                               ExpDirPrinterCallback(EXP_PATH),
                               # Checkpointer(EXP_PATH),
                               loss_scheduler,
                              ])

    # test one input
    # @ignore_derivatives model(FastAI.getbatch(learner)[1] |> DEVICE)
    n_epochs=(DRY ? 200 : 1000)
    fit!(learner, n_epochs, (VAETrainingPhase()=>dl, 
                             VAEValidationPhase()=>dl_val))
    # for _ in 1:4
    #     fitonecycle!(learner, 20, 3e-4;
    #                  phases=(VAETrainingPhase()   => dl,
    #                          VAEValidationPhase() => dl_val))
    # end
    # close(wandb_backend)
    # model_cpu = cpu(model);
    # @save joinpath(EXP_PATH, "model_ep_$n_epochs.bson") model_cpu
    # savetaskmodel(joinpath(EXP_PATH, "model_ep_$n_epochs.jld2"), task, learner.model)
    #####################################################
end
