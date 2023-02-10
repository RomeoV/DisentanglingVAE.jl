import StatsBase: mean
import Random: randn!
using FastAI
using FastAI.FluxTraining
using FastAI.Flux
import FastAI.Flux.Losses: logitbinarycrossentropy
using FastAI: encodedblock, encodedblockfilled, decodedblockfilled

using FastVision
import FastVision: RGB
using Metalhead
# using FastTimm
# using PyCall, PyNNTraining

import FastAI.Flux.ChainRulesCore: @ignore_derivatives
import FastAI.Flux.MLUtils: _default_executor
import FastAI.MLUtils.Transducers: ThreadedEx
# ThreadPoolEx gave me problems, see https://github.com/JuliaML/MLUtils.jl/issues/142
_default_executor() = ThreadedEx()
DEVICE = gpu
BE = ShowText();  # sometimes get segfault by default

#### Prepare dataset and task ######################################
# path = load(datasets()["mnist_tiny"])
path = load(datasets()["CUB_200_2011"])
data = Datasets.loadfolderdata(
    path,
    filterfn = FastVision.isimagefile,
    loadfn = loadfile,
)
# data, blocks = load(datarecipes()["CUB_200_2011"])
showblock(BE, Image{2}(), getobs(data, rand(1:numobs(data))))

function EmbeddingTask(block, enc)
  sample = block
  encodedsample = x = y = ŷ = encodedblockfilled(enc, block)
  blocks = (; sample, x, y, ŷ, encodedsample)
  return BlockTask(blocks, enc)
end

# Note: I checked and the imagenet means and stds are roughly applicable.
task = EmbeddingTask(Image{2}(),
                     (ProjectiveTransforms((64, 64)),
                      ImagePreprocessing(means=FastVision.SVector(0., 0., 0.),
                                         stds=FastVision.SVector(1., 1., 1.);
                                         C = RGB{Float32},
                                         buffered=false,
                                        ),
                     )
                    )

# BATCHSIZE=2
BATCHSIZE=8
dl, dl_val = taskdataloaders(data, task, BATCHSIZE, pctgval=0.1;
                             buffer=true, partial=false);
####################################################################

##### Set up VAE ##########
struct VAE{E, B, D}
  encoder::E
  bridge ::B
  decoder::D
end
Flux.@functor VAE

function (vae::VAE)(x)
  intermediate = vae.encoder(x)
  μ, logσ² = vae.bridge(intermediate)
  z = sample_latent(μ, logσ²)
  x̄ = vae.decoder(z)
  return x̄, (; μ, logσ²)
end

sample_latent(μ::AbstractArray{T}, logσ²::AbstractArray{T}) where {T} =
       μ .+ exp.(logσ²./2) .* randn!(similar(logσ²))

bernoulli_loss(x, x_rec) = logitbinarycrossentropy(x_rec, x;
                                                   agg=x->sum(x; dims=[1,2,3]))
function ELBO(x, x̄, μ, logσ²)
  # reconstruction_error = mean(sum(@. ((x̄ - x)^2); dims=(1,2,3)))
  reconstruction_error = bernoulli_loss(x, x̄)
  kl_divergence = sum(@. ((μ^2 + exp(logσ²) - 1 - logσ²) / 2); dims=1)
  return mean(reconstruction_error) + mean(kl_divergence)
end
########################

#### Set up model #########
# image size is (64, 64)
backbone_dim = 512
latent_dim = 64

backbone = Metalhead.ResNet(18; pretrain=true)
backbone = Chain(backbone.layers[1], Chain(backbone.layers[2].layers[1:2]..., identity)) |> DEVICE

bridge = 
Chain(Dense(backbone_dim, backbone_dim, leakyrelu),
      LayerNorm(backbone_dim),
      Parallel(
          tuple,
          # Special initialization, see https://arxiv.org/pdf/2010.14407.pdf, Table 2 (Appendix)
          Dense(0.1f0*Flux.glorot_uniform(latent_dim, backbone_dim),
                -1*ones(Float32, latent_dim)),  # μ
          Dense(0.1f0*Flux.glorot_uniform(latent_dim, backbone_dim),
                -1*ones(Float32, latent_dim)),  # logvar
         )
     ) |> DEVICE

ResidualBlock(c) = Parallel(+,
                            Chain(leakyrelu,
                                  Conv((3, 3), c=>c, identity; pad=SamePad()), 
                                  leakyrelu,
                                  Conv((3, 3), c=>c, identity; pad=SamePad())),
                            identity)

decoder = Chain(Dense(latent_dim, 4*4*64, leakyrelu),
                Dense(4*4*64, 4*4*256, identity),
                xs -> reshape(xs, 4, 4, 256, :),
                ResidualBlock(256),
                ResidualBlock(256),
                Upsample(2),
                ResidualBlock(256),
                ResidualBlock(256),
                Conv((1, 1), 256=>128, identity),
                ResidualBlock(128),
                ResidualBlock(128),
                Upsample(2),
                ResidualBlock(128),
                ResidualBlock(128),
                Conv((1, 1), 128=>64, identity),
                Upsample(2),
                ResidualBlock(64),
                ResidualBlock(64),
                Upsample(2),
                leakyrelu,
                Conv((5, 5), 64=>3; pad=SamePad())) |> DEVICE


# decoder = Chain(Dense(latent_dim, 4*4*64,  leakyrelu),
#                 Dense(4*4*64,     4*4*256, identity),
#                 xs -> reshape(xs, 4, 4, 256, :),
#                 Parallel(addrelu,
#                   basicblock(256, (256, 256)),
#                   identity),
#                 Parallel(addrelu,
#                   basicblock(256, (256, 256)),
#                   identity),
#                 PixelShuffle(2),
#                 Upsample(2),
#                 Parallel(addrelu,
#                   basicblock(64, (64, 64)),
#                   identity),
#                 Parallel(addrelu,
#                   basicblock(64, (64, 64)),
#                   identity),
#                 PixelShuffle(2),
#                 Upsample(2),
#                 Conv((1, 1), 16=>3),
#                 sigmoid
#                ) |> DEVICE

model = VAE(backbone, bridge, decoder);
params = Flux.params(bridge, decoder);
###########################

#### Use FluxTraining to define how this model is run. #####
struct VAETrainingPhase <: FluxTraining.AbstractTrainingPhase end
struct VAEValidationPhase <: FluxTraining.AbstractValidationPhase end
function FluxTraining.on(
    ::FluxTraining.StepBegin,
    ::Union{VAETrainingPhase, VAEValidationPhase},
    cb::ToDevice,
    learner,
  )
  learner.step.x = cb.movedatafn(learner.step.x)
end

function FluxTraining.step!(learner, phase::VAETrainingPhase, batch)
  FluxTraining.runstep(learner, phase, (x = batch,)) do handle, state
    gs = gradient(params) do
      intermediate   = @ignore_derivatives learner.model.encoder(state.x)
      μ, logσ²       = learner.model.bridge(intermediate)
      state.z        = sample_latent(μ, logσ²)
      state.x̄        = learner.model.decoder(state.z)

      handle(FluxTraining.LossBegin())
      state.loss = learner.lossfn(state.x, state.x̄, μ, logσ²)

      handle(FluxTraining.BackwardBegin())
      return state.loss
    end
    handle(FluxTraining.BackwardEnd())
    Flux.Optimise.update!(learner.optimizer, params, gs)
  end
end

function FluxTraining.step!(learner, phase::VAEValidationPhase, batch)
  FluxTraining.runstep(learner, phase, (x = batch,)) do handle, state
    @ignore_derivatives begin
      intermediate   = learner.model.encoder(state.x)
      μ, logσ²       = learner.model.bridge(intermediate)
      state.z        = sample_latent(μ, logσ²)
      state.x̄        = learner.model.decoder(state.z)
      state.loss = learner.lossfn(state.x, state.x̄, μ, logσ²)
    end
  end
end
############################################################

#### Try to run the training. #######################
opt = Flux.Optimiser(ClipNorm(1), Adam())
# opt = Flux.Optimiser(Adam())
learner = Learner(model, ELBO;
                  optimizer=opt,
                  data=(dl, dl_val),
                  callbacks=[ToGPU(), ProgressPrinter()])

# test one input
@ignore_derivatives model(getbatch(learner) |> gpu)[1] |> size
fitonecycle!(learner, 5, 1e-4; phases=(VAETrainingPhase() => dl,
                                        VAEValidationPhase() => dl_val))
#####################################################

xs = makebatch(task, data, rand(1:numobs(data), 4)) |> DEVICE;
ypreds, _ = model(xs);
showoutputbatch(BE, task, cpu(xs), cpu(ypreds) .|> sigmoid)
