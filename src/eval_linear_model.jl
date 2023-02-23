using DisentanglingVAE
using GLM
using DataFrames
using Distributions
using BSON
using FastAI.Flux
import FastAI.Flux.MLUtils
import Metalhead
import FastAI.Flux: cpu
using FastAI: mapobs, taskdataloaders
import BSON: @load
using StaticArrays
using Plots, StatsPlots

import FastAI.Flux.MLUtils: _default_executor
import FastAI.MLUtils.Transducers: ThreadedEx
_default_executor() = ThreadedEx()
DEVICE = cpu;

# make data
# line_data = mapobs(DisentanglingVAE.make_data_sample, 1:2^13)
line_data = mapobs(DisentanglingVAE.make_data_sample_uniform, 1:2^13)

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
  intermediate = model.encoder(img_lhs)
  μ, logσ² = model.bridge(intermediate)
  # distribution_transform(z) = if true
  #   # use this if distribution is origianally uniform
  #   z->cdf(Normal(0, 1), z)
  # else
  #   # use this if distribution is origianally normal
  #   z->z
  # end
  distribution_transform(z) = cdf(Normal(0, 1), z)
  y_pre = distribution_transform.(μ)
  σ_pre = distribution_transform'.(μ) .* exp.(logσ²/2)
  #
  for i in axes(y_pre)[end]
    push!(df, Dict(:y_raw=>μ[:, i],
                   :y_pre=>y_pre[:, i],
                   :σ_pre=>σ_pre[:, i],
                   :y_gt =>v_lhs[:, i]))
  end
end

# process df
data = copy(df)
data.y_gt_1  = getindex.(data.y_gt,  1)
data.y_gt_2  = getindex.(data.y_gt,  2)
data.y_gt_3  = getindex.(data.y_gt,  3)
data.y_gt_4  = getindex.(data.y_gt,  4)
data.y_gt_5  = getindex.(data.y_gt,  5)
data.y_gt_6  = getindex.(data.y_gt,  6)
data.y_raw_1 = getindex.(data.y_raw, 1)
data.y_raw_2 = getindex.(data.y_raw, 2)
data.y_raw_3 = getindex.(data.y_raw, 3)
data.y_raw_4 = getindex.(data.y_raw, 4)
data.y_raw_5 = getindex.(data.y_raw, 5)
data.y_raw_6 = getindex.(data.y_raw, 6)
data.y_pre_1 = getindex.(data.y_pre, 1)
data.y_pre_2 = getindex.(data.y_pre, 2)
data.y_pre_3 = getindex.(data.y_pre, 3)
data.y_pre_4 = getindex.(data.y_pre, 4)
data.y_pre_5 = getindex.(data.y_pre, 5)
data.y_pre_6 = getindex.(data.y_pre, 6)

# train linear model
model_1 = lm(@formula(y_gt_1 ~ 1 + y_pre_1 + y_pre_2 + y_pre_3 + y_pre_4 + y_pre_5 + y_pre_6), data)
data.y_pred_1 = predict(model_1);

model_2 = lm(@formula(y_gt_2 ~ 1 + y_pre_1 + y_pre_2 + y_pre_3 + y_pre_4 + y_pre_5 + y_pre_6), data)
data.y_pred_2 = predict(model_2);

model_3 = lm(@formula(y_gt_3 ~ 1 + y_pre_1 + y_pre_2 + y_pre_3 + y_pre_4 + y_pre_5 + y_pre_6), data)
data.y_pred_3 = predict(model_3);
model_4 = lm(@formula(y_gt_4 ~ 1 + y_pre_1 + y_pre_2 + y_pre_3 + y_pre_4 + y_pre_5 + y_pre_6), data)
data.y_pred_4 = predict(model_4);
model_5 = lm(@formula(y_gt_5 ~ 1 + y_pre_1 + y_pre_2 + y_pre_3 + y_pre_4 + y_pre_5 + y_pre_6), data)
data.y_pred_5 = predict(model_5);
model_6 = lm(@formula(y_gt_6 ~ 1 + y_pre_1 + y_pre_2 + y_pre_3 + y_pre_4 + y_pre_5 + y_pre_6), data)
data.y_pred_6 = predict(model_6);

plt = plot(; title="Density of embedding means. Should be \$\\mathcal{N}(0, 1)\$.")
for i in 1:6
  density!(plt, data[!, "y_raw_$i"])
end
plt

plt = begin
  plt1 = plot()
  @df data density!(:y_gt_1, title="Dimension 1")
  @df data density!(:y_pred_1)
  plt2 = plot()
  @df data density!(:y_gt_2, title="Dimension 2")
  @df data density!(:y_pred_2)
  plt3 = plot()
  @df data density!(:y_gt_3, title="Dimension 3")
  @df data density!(:y_pred_3)
  plt4 = plot()
  @df data density!(:y_gt_4, title="Dimension 4")
  @df data density!(:y_pred_4)
  plt5 = plot()
  @df data density!(:y_gt_5, title="Dimension 5")
  @df data density!(:y_pred_5)
  plt6 = plot()
  @df data density!(:y_gt_6, title="Dimension 6")
  @df data density!(:y_pred_6)
  plot(plt1, plt2, plt3, plt4, plt5, plt6)
end
