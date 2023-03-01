import FastAI
import FluxTraining
import FluxTraining: Read, Write
import GLM
import Printf: @sprintf
using EllipsisNotation
import InvertedIndices: Not
import ChainRulesCore: @ignore_derivatives
import OrderedCollections: OrderedDict

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
  GC.gc()
end
FluxTraining.stateaccess(::VisualizationCallback) = (data=FluxTraining.Read(), 
                                                                      model=FluxTraining.Read(), )
runafter(::VisualizationCallback) = (Metrics,)


struct LinearModelCallback <: FluxTraining.Callback
    device
end

function FluxTraining.on(
                ::FluxTraining.EpochEnd,
                phase::FluxTraining.Phases.AbstractValidationPhase,
                cb::LinearModelCallback,
                learner)
    encoder, bridge = learner.model.encoder, learner.model.bridge

    # collect data
    predictors = Array{Float64, 2}[];
    labels     = Array{Float64, 2}[];
    for (xs, ys, _, _, _) in learner.data[:validation]
        μs = @ignore_derivatives (bridge ∘ encoder)(xs |> cb.device)[1] |> cpu
        push!(predictors, μs')
        push!(labels, ys')
    end
    predictors = let predictors = cat(predictors..., dims=1)
        [predictors ;; ones(size(predictors)[1])]  # we add a bias here
    end
    labels = cat(labels..., dims=1)

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
end
FluxTraining.stateaccess(::LinearModelCallback) = (data=Read(),
                                                   model=Read(),
                                                   cbstate=(metricsepoch=Write(), history=Read()),)
FluxTraining.resolveconflict(::FluxTraining.Metrics, ::LinearModelCallback) = FluxTraining.NoConflict()
FluxTraining.resolveconflict(::LinearModelCallback, ::FluxTraining.Metrics) = FluxTraining.NoConflict()
runafter(::LinearModelCallback) = (VisualizationCallback,)
