import FastAI, FastAI.FluxTraining
import FastAI.FluxTraining: on
using EllipsisNotation

struct VisualizationCallback <: FluxTraining.Callback 
  task
  device
end

function FluxTraining.on(
                ::FluxTraining.EpochEnd,
                ::FluxTraining.Phases.AbstractValidationPhase,
                cb::DisentanglingVAE.VisualizationCallback,
                learner)
  xs = first(learner.data[:validation]) .|> x->x[.., 1:1] |> cb.device
  ys = learner.model(xs; apply_sigmoid=true);
  FastAI.showoutputbatch(ShowText(), cb.task, cpu.(xs), cpu.(ys))
  GC.gc()
end
FluxTraining.stateaccess(::DisentanglingVAE.VisualizationCallback) = (data=FluxTraining.Read(), 
                                                                      model=FluxTraining.Read(), )
runafter(::VisualizationCallback) = (Metrics,)
