import FastAI
import FastAI: ShowText
import FluxTraining
import FluxTraining: Read, Write, Loggables, _combinename
import GLM
import Printf: @sprintf
using EllipsisNotation
import InvertedIndices: Not
import ChainRulesCore: @ignore_derivatives
import OrderedCollections: OrderedDict
import Flux: cpu, gpu
import FluxTraining: LoggerBackend
import CSV
import DataFrames
import DataFrames: DataFrame, nrow

struct VisualizationCallback <: FluxTraining.Callback 
  task
  device
end

function FluxTraining.on(
                ::FluxTraining.EpochEnd,
                ::FluxTraining.Phases.AbstractValidationPhase,
                cb::VisualizationCallback,
                learner)
  xs = first(learner.data[:validation]) .|> x->x[.., 1:1] |> cb.device
  ys = learner.model(xs; apply_sigmoid=true);
  FastAI.showoutputbatch(ShowText(), cb.task, cpu.(xs), cpu.(ys))
  # GC.gc()
end
FluxTraining.stateaccess(::VisualizationCallback) = (data=FluxTraining.Read(), 
                                                     model=FluxTraining.Read(), )
FluxTraining.runafter(::VisualizationCallback) = (Metrics,)


struct LinearModelCallback <: FluxTraining.Callback
    device
end

function FluxTraining.on(
                ::FluxTraining.EpochEnd,
                phase::FluxTraining.Phases.AbstractTrainingPhase,
                cb::LinearModelCallback,
                learner)
    epoch = learner.cbstate.history[phase].epochs
    if epoch % 10 == 0
        eval_lin_callback_with_data(phase, cb, learner, :training)
    end
end
function FluxTraining.on(
                ::FluxTraining.EpochEnd,
                phase::FluxTraining.Phases.AbstractValidationPhase,
                cb::LinearModelCallback,
                learner)
    epoch = learner.cbstate.history[phase].epochs
    if epoch % 10 != 0
        eval_lin_callback_with_data(phase, cb, learner, :validation)
    end
end
function eval_lin_callback_with_data(
        phase::Union{FluxTraining.Phases.AbstractTrainingPhase,
                     FluxTraining.Phases.AbstractValidationPhase},
        cb::LinearModelCallback,
        learner,
        dset_symbol
    )
    encoder, bridge = learner.model.encoder, learner.model.bridge

    # collect data
    # after combining batches, the final result will be in nobs x npredictors
    predictors    = Array{Float64, 2}[];
    labels        = Array{Float64, 2}[];
    uncertainties = Array{Float64, 2}[];
    for (xs, ys, _, _, _) in learner.data[dset_symbol]
        μs, logσ²s = @ignore_derivatives (bridge ∘ encoder)(xs |> cb.device) .|> cpu
        push!(predictors, μs')
        push!(uncertainties, exp.(logσ²s./2)')
        push!(labels, ys')
    end
    predictors = let predictors = vcat(predictors...)
        hcat(predictors, ones(size(predictors, 1)))  # we add a bias here
    end
    labels        = vcat(labels...)
    uncertainties = vcat(labels...)

    # linear model
    models = OrderedDict(
        i => GLM.lm(predictors, ys)
        for (i, ys) in enumerate(eachslice(labels, dims=2))
    )
    # print pvals
    p_vals(i) = GLM.coeftable(models[i]).cols[4]
    println("p values (bias last)")
    display(cat([p_vals(i)'.|> x->@sprintf("%.3e", x)
                 for i in keys(models)]..., dims=1))

    # Log "correct" p value and next highest confidence one as metric value
    # In a graph we should see that the correct value goes to zero
    # and the next highest one goes up
    epoch = learner.cbstate.history[phase].epochs
    metricsepoch = learner.cbstate.metricsepoch[phase]
    for (i, model) in models
        p_vals = GLM.coeftable(model).cols[4]
        p_ast  = p_vals[i]  # p value of correct predictor
        p_next = minimum(p_vals[Not(1, i)])  # next "highest confidence" p value
        push!(metricsepoch, Symbol("p$(i)*"), epoch, p_ast)
        push!(metricsepoch, Symbol("p$(i)_"), epoch, p_next)
    end

    ## UNCERTAINTY QUANTIFICATION
    # This is hacky, but I want to reuse the computed predictors also for the uncertainty quantification.
    # Probably the better way would be to store the predictors once in another callback state, and then use them from another callback.
    # Maybe tomorrow ;)
    for (i, (μs, σs, ys)) in enumerate(zip(eachslice(predictors,    dims=2),
                                           eachslice(uncertainties, dims=2),
                                           eachslice(labels,        dims=2)))
        c_min, c_max, c_mean = compute_calibration_metric(
            Normal.(μs, σs), ys
        )
        push!(metricsepoch, Symbol("c$(i)_min"),  epoch, c_min)
        push!(metricsepoch, Symbol("c$(i)_max"),  epoch, c_max)
        push!(metricsepoch, Symbol("c$(i)_mean"), epoch, c_mean)

        dispersion = compute_dispersion(
            Normal.(μs, σs), ys
        )
        push!(metricsepoch, Symbol("d$(i)"), epoch, dispersion)
    end
end
FluxTraining.stateaccess(::LinearModelCallback) = (data=Read(),
                                                   model=Read(),
                                                   cbstate=(metricsepoch=Write(), history=Read()),)
FluxTraining.runafter(::LinearModelCallback) = (VisualizationCallback, Metrics, )

struct ExpDirPrinterCallback <: FluxTraining.Callback
    path
end
function FluxTraining.on(
                ::FluxTraining.EpochBegin,
                ::FluxTraining.Phases.AbstractTrainingPhase,
                cb::ExpDirPrinterCallback,
                learner)
    println(cb.path)
end
FluxTraining.stateaccess(::ExpDirPrinterCallback) = (; )

struct CSVLoggerBackend_ <: LoggerBackend
    logdir :: String
    df :: DataFrames.DataFrame
    function CSVLoggerBackend_(logdir, n_vars)
        names = [["Loss"];
                 ["p$(i)*"     for i in 1:n_vars];  # linear model p-value of true predictor
                 ["p$(i)_"     for i in 1:n_vars];  # minimum linear model p-value of false predictors
                 ["c$(i)_min"  for i in 1:n_vars];  # minimum calibration error
                 ["c$(i)_max"  for i in 1:n_vars];  # maximum calibration error
                 ["c$(i)_mean" for i in 1:n_vars];  # mean calibration error
                 ["d$(i)"      for i in 1:n_vars]]  # dispersion
        arrs = fill(Float64[], length(names))
        df = DataFrame(arrs, names)
        new(logdir, df)
    end
end
CSVLoggerBackend = CSVLoggerBackend_

Base.show(io::IO, backend::CSVLoggerBackend) = print(
    io, "CSVLoggerBackend(", backend.logdir, ")")

function log_to(backend::CSVLoggerBackend, value::Loggables.Value, name, i; group = ())
    if nrow(backend.df) < i
        push!(backend.df, fill(NaN, length(names(backend.df))))
    end

    backend.df[i, name] = value.data
    CSV.write(joinpath(backend.logdir, "log.csv"), backend.df)
end
