using FastAI: Continuous, LabelMulti, Encoding
import FastAI: encode, encodedblock
using FastVision: RGB, Image, ImageToTensor
using FastVision.DataAugmentation: apply
import FastVision.DataAugmentation
using ImageCore: colorview
"""
- `blocks.sample`: The most important block, representing one full
    observation of unprocessed data. Data containers used with a learning
    task should have compatible observations, i.e.
    `checkblock(blocks.sample, data[i])`.
- `blocks.x`: Data that will be fed into the model, i.e. (neglecting batching)
    `model(x)` should work
- `blocks.ŷ`: Data that is output by the model, i.e. (neglecting batching)
    `checkblock(blocks.ŷ, model(x))`
- `blocks.y`: Data that is compared to the model output using a loss function,
    i.e. `lossfn(ŷ, y)`
- `blocks.encodedsample`: An encoded version of `blocks.sample`. Will usually
    correspond to `encodedblockfilled(getencodings(task), blocks.sample)`.
"""

# task = DisentanglingVAETask((Continuous(6), Continuous(6), LabelMulti(1:6)), LineTupleEncoding())

# encoding input output
InputBlockTuple = Tuple{Continuous, Continuous, LabelMulti}
InputBlockTuple_(N::Int) = (Continuous(N), Continuous(N), LabelMulti(1:N))
OutputBlockTuple = Tuple{Image{2}, Continuous, Image{2}, Continuous, LabelMulti}
OutputBlockTuple_(N) = (Image{2}(), Continuous(N), Image{2}(), Continuous(N), LabelMulti(1:N))

# sample: tuple of (v_lhs, v_rhs, ks)
# x, x_lhs, x_rhs: Image{2}
# y = x
# x̄, x̄_lhs, x_rhs: Image{2}
# x̄, x̄_lhs, x_rhs: Image{2}
# y\hat = x\bar
# \mu_lhs, \mu_rhs, log\sigma\^2_lhs, log\sigma\^2_lhs
# enc :: Tuple(Continuous, Continuous, LabelMulti) -> Tuple(Image, Continuous, Image, Continuous, LabelMulti)
function DisentanglingVAETask(in_blocks::InputBlockTuple, enc)
  sample = in_blocks
  x_lhs, x_rhs, ks = sample
  encodedsample = encodedblock(enc, sample)
  N = length(ks.classes)
  x, y, ŷ = x_lhs, x_lhs, x_lhs 
  μ_lhs, μ_rhs = Continuous(N), Continuous(N)
  logσ²_lhs, logσ²_rhs = Continuous(N), Continuous(N)

  blocks = (; sample, x, y, ŷ, encodedsample, μ_lhs, μ_rhs, logσ²_lhs, logσ²_rhs)
  return BlockTask(blocks, enc)
end


"""
- `encode(::E, ::Context, block::Block, data)` encodes `block` of `data`.
    The default is to do nothing. This should be overloaded for an encoding `E`,
    concrete `Block` types and possibly a context.
- `encodedblock(::E, block::Block) -> block'` returns the block that is obtained by
    encoding `block` with encoding `E`. This needs to be constant for an instance of `E`,
    so it cannot depend on the sample or on randomness. The default is to return `nothing`,
    meaning the same block is returned and not changed. Encodings that return the same
    block but change the data (e.g. `ProjectiveTransforms`) should return `block`.
"""
struct LineTupleEncoding <: Encoding end

encodedblock(::LineTupleEncoding, block::InputBlockTuple) = OutputBlockTuple_(block[1].size)

function encode(::LineTupleEncoding, ::Context, block::InputBlockTuple, obs::NamedTuple)
  v_lhs, v_rhs, ks = obs
  img_lhs, img_rhs = zeros(Float32, 3, 28, 28), zeros(Float32, 3, 28, 28)
  draw_labels_on_image_two_ways!((img_lhs, img_rhs), (v_lhs, v_rhs))
  img_lhs = apply(ImageToTensor(), DataAugmentation.Image(colorview(RGB, img_lhs))) |> DataAugmentation.itemdata
  img_rhs = apply(ImageToTensor(), DataAugmentation.Image(colorview(RGB, img_rhs))) |> DataAugmentation.itemdata
  return (img_lhs, v_lhs, img_rhs, v_rhs, ks)
end


# encode(getencodings(task), Training(), getblocks(task).sample, inp_data)
