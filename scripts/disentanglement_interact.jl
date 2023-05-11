using DisentanglingVAE
import Distributions: make_data_sample
using DataFrames
using BSON
using Distributions
using FastAI: Continuous
using FastAI.Flux
using StaticArrays: SVector
import FastAI.Flux.MLUtils
import FastAI.Flux: cpu
using FastAI: mapobs, taskdataloaders, ObsView, datarecipes, load
import BSON: @load
using GLMakie
using Makie
using Images
import MLUtils
using EllipsisNotation

import FastAI.Flux.MLUtils: _default_executor
import FastAI.MLUtils.Transducers: ThreadedEx
_default_executor() = ThreadedEx()
DEVICE = cpu;


line_data = mapobs(make_data_sample, 1:2^10)

task = DisentanglingVAETask()
BATCHSIZE=128
dl, dl_val = taskdataloaders(line_data, line_data, task, BATCHSIZE;
                             shuffle=false, parallel=false);

# we can also use the dataloader directly, but it doesn't apply the task transforms.
# probably we could use `encodesample` or something like that.
# dl = MLUtils.DataLoader(line_data; batchsize=BATCHSIZE, collate=true, shuffle=false)

model = begin
    @load "/home/romeo/Documents/Stanford/google_ood/DisentanglingVAE.jl/experiments/directionality-loss_89a7c31_4776eb46/version_1/checkpoint_epoch_090_loss_2107.814719063895.bson" model;
    model
end

data = (isdefined(Main, :data) ? data : begin
    IND = 1:6
    df = DataFrame([SVector{6, Float64}[],
                    SVector{6, Float64}[],
                    SVector{6, Float64}[]], [:z_raw, :σ_raw, :v_gt])
    for (img_lhs, v_lhs, img_rhs, v_rhs, ks) in dl
        for (img, v) in [(img_lhs, v_lhs), ] #(img_rhs, v_rhs)]
            intermediate = model.encoder(img)
            μ, logσ² = model.bridge(intermediate)
            for i in axes(μ)[end]
            push!(df, (; :z_raw=>μ[:, i][IND],
                        :σ_raw=>exp.(logσ²[:, i][IND]./2),
                        :v_gt =>v[:, i]))
            end
        end
    end

    # process df
    v_gt_syms  = [Symbol("v_gt_$i")  for i in IND]
    z_raw_syms = [Symbol("z_raw_$i") for i in IND]
    σ_raw_syms = [Symbol("σ_raw_$i") for i in IND]
    data = transform(df, :v_gt =>v_gt_syms,
                        :z_raw=>z_raw_syms,
                        :σ_raw=>σ_raw_syms)
    data
end)

## Start visualization section.

fig = Figure()
ls_ij = SliderGrid(fig[2, 1],
                   (label="i", range=1:6, startvalue=1),
                   (label="j", range=1:6, startvalue=1),
                   # (label="idx", range=1:300, startvalue=1);
                   tellwidth=false
                   )

## scatter plot for points
scatter_data_x = lift(ls_ij.sliders[1].value, ls_ij.sliders[2].value) do i, j
    data[!, "v_gt_$i"]
end
scatter_data_y = lift(ls_ij.sliders[1].value, ls_ij.sliders[2].value) do i, j
    data[!, "z_raw_$j"]
end
img_lhs = scatter(fig[1, 1], scatter_data_x, scatter_data_y;
                  axis=(; xlabel="v_gt_i", ylabel="z_raw_j",
                        limits=((-2.5, 2.5), (-2.5, 2.5))))

idx = Observable{UInt32}(0x1)
function Makie.show_data(inspector::DataInspector, plot::Scatter, idx_::UInt32)
    # notice that for some reason we get `idx_ = 1` all the time, idk why.
    # we want to ignore those, so we just check for > 1.
    if idx_ > 1
        idx[] = idx_
        notify(idx)
    end

    return true
end

# util, can also be found in IterTools.jl
function nth(xs, n)
    for (i, val) in enumerate(xs)
        i >= n && return val
    end
end

## plot correct image
img_plt = lift(idx) do idx
    img_arr = nth(dl, idx÷BATCHSIZE + 1)[1][:, :, :, idx % BATCHSIZE] |> x->permutedims(x, (3, 1, 2))
    img_arr .|> float64 |> colorview(RGB)
end
image(fig[1, 2], img_plt)

## highlight currently selected datapoint
highlight_x = lift(ls_ij.sliders[1].value, ls_ij.sliders[2].value, idx) do i, j, idx
    data[idx:idx, "v_gt_$i"]
end
highlight_y = lift(ls_ij.sliders[1].value, ls_ij.sliders[2].value, idx) do i, j, idx
    data[idx:idx, "z_raw_$j"]
end
scatter!(fig[1, 1], highlight_x, highlight_y; color=:red)

DataInspector(fig)

fig
