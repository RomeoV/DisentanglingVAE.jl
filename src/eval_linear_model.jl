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

IND = 1:6
df = DataFrame([SVector{6, Float64}[],
                SVector{6, Float64}[],
                SVector{6, Float64}[]], [:y_raw, :σ_raw, :y_gt])
for (img_lhs, v_lhs, img_rhs, v_rhs, ks) in dl
  for (img, v) in [(img_lhs, v_lhs), (img_rhs, v_rhs)]
    intermediate = model.encoder(img)
    μ, logσ² = model.bridge(intermediate)
    for i in axes(μ)[end]
      push!(df, (; :y_raw=>μ[:, i],
                   :σ_raw=>exp.(logσ²[:, i]./2),
                   :y_gt =>v[:, i]))
    end
  end
end

# process df
y_gt_syms = [Symbol("y_gt_$i") for i in IND]
y_raw_syms = [Symbol("y_raw_$i") for i in IND]
y_pre_syms = [Symbol("y_pre_$i") for i in IND]
σ_raw_syms = [Symbol("σ_raw_$i") for i in IND]
σ_pre_syms = [Symbol("σ_pre_$i") for i in IND]
data = transform(df, :y_gt=>y_gt_syms, :y_raw=>y_raw_syms, :σ_raw=>σ_raw_syms)

# transform to uniform by fitting a gaussian on the marginal
for i in IND
  D = Distributions.fit_mle(Normal, data[!, y_raw_syms[i]])
  T(z) = cdf(D, z)  # transform marginal distribution to uniform
  # T(z) = z
  data[!, y_pre_syms[i]] = T.(data[!, y_raw_syms[i]])
  data[!, σ_pre_syms[i]] = inv.(T'.(data[!, y_raw_syms[i]])) .* data[!, σ_raw_syms[i]]
end

# train linear model
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
