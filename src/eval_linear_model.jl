using DisentanglingVAE
using GLM
using DataFrames
using Distributions
using BSON
using FastAI: Continuous
using FastAI.Flux
import FastAI.Flux.MLUtils
import Metalhead
import FastAI.Flux: cpu
using FastAI: mapobs, taskdataloaders, ObsView
import BSON: @load
using StaticArrays
using Plots, StatsPlots

import FastAI.Flux.MLUtils: _default_executor
import FastAI.MLUtils.Transducers: ThreadedEx
_default_executor() = ThreadedEx()
DEVICE = cpu;

# make data
# line_data = mapobs(DisentanglingVAE.make_data_sample, 1:2^13)
data_fn(i) = DisentanglingVAE.make_data_sample(Uniform, i)
line_data = mapobs(data_fn, ObsView(1:2^13))

task = DisentanglingVAETask()

BATCHSIZE=128
dl, dl_val = taskdataloaders(line_data, task, BATCHSIZE, pctgval=0.1;
                             buffer=false, partial=false,
                            );

@load "models/model.bson" model_cpu;
model = model_cpu

df = DataFrame([SVector{6, Float64}[],
                SVector{6, Float64}[],
                SVector{6, Float64}[],
                SVector{6, Float64}[]], [:y_raw, :y_pre, :σ_pre, :y_gt])
for (img_lhs, v_lhs, img_rhs, v_rhs, ks) in dl
  for img in [img_lhs, img_rhs]
    intermediate = model.encoder(img)
    μ, logσ² = model.bridge(intermediate)
    distribution_transform(z) = cdf(Normal(0, 1), z)
    # distribution_transform(z) = identity(z)
    y_pre = distribution_transform.(μ)
    σ_pre = inv.(distribution_transform'.(μ)) .* exp.(logσ²/2)
    #
    for i in axes(y_pre)[end]
      push!(df, Dict(:y_raw=>μ[:, i],
                     :y_pre=>y_pre[:, i],
                     :σ_pre=>σ_pre[:, i],
                     :y_gt =>v_lhs[:, i]))
    end
  end
end

# process df
y_gt_syms = [Symbol("y_gt_$i") for i in 1:6]
y_raw_syms = [Symbol("y_raw_$i") for i in 1:6]
y_pre_syms = [Symbol("y_pre_$i") for i in 1:6]
data = transform(df, :y_gt=>y_gt_syms, :y_raw=>y_raw_syms, :y_pre=>y_pre_syms)



# train linear model
IND = 1:6
model = Dict(
             i => lm(Term(y_gt_syms[i]) ~ term(1)+sum(Term.(y_pre_syms)), data)
             for i in IND
            )
for i in IND
  data[!, "y_pred_$i"] = predict(model[i])
end

plt = begin
  plt = plot(; title="Density of embedding means. Should be \$\\mathcal{N}(0, 1)\$.")
  for i in 1:6
    density!(plt, data[!, "y_raw_$i"])
  end
  plt
end

plt = begin
  plts = [begin
            plt = plot()
            # the syntax here is a bit annoying (cols(Symbol(...)))
            # see https://github.com/JuliaPlots/StatsPlots.jl/issues/316
            @df data density!(cols(Symbol("y_gt_$i")), title="Dimension $i", label="label")
            @df data density!(cols(Symbol("y_pred_$i")), label="estimate")
          end for i in IND]
  plot(plts...; layout=(6, 1))
end
