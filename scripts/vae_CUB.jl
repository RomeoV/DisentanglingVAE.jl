using Statistics: mean
using FastAI
using FastAI.FluxTraining
using FastAI.Flux
using FastAI: encodedblock, encodedblockfilled, decodedblockfilled
import FastAI.Flux.MLUtils: _default_executor
import FastAI.MLUtils.Transducers: ThreadedEx
using FastVision
import FastVision: RGB
using FastTimm
using Metalhead
using PyCall, PyNNTraining
# ThreadPoolEx gave me problems, see https://github.com/JuliaML/MLUtils.jl/issues/142
_default_executor() = ThreadedEx()
DEVICE = gpu
BE = ShowText();  # sometimes get segfault by default

#### Prepare dataset and task ######################################
# path = load(datasets()["CUB_200_2011"])
path = load(datasets()["mnist_tiny"])
data = Datasets.loadfolderdata(
    path,
    filterfn = FastVision.isimagefile,
    loadfn = loadfile,
)
# data, blocks = load(datarecipes()["CUB_200_2011"])
# showblock(BE, Image{2}(), getobs(data, 3))

function EmbeddingTask(block, enc)
  sample = block
  encodedsample = x = y = ŷ = encodedblockfilled(enc, block)
  blocks = (; sample, x, y, ŷ, encodedsample)
  return BlockTask(blocks, enc)
end

# Note: I checked and the imagenet means and stds are roughly applicable.
task = EmbeddingTask(Image{2}(),
                     (ProjectiveTransforms((64, 64)),
                      ImagePreprocessing(means = FastVision.IMAGENET_MEANS,
                                         stds = FastVision.IMAGENET_STDS,
                                         C = RGB{Float32},
                                         buffered=false,
                                        ),
                     )
                    )

BATCHSIZE=2
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

function ELBO(x, x̄, μ, logσ²)
  reconstruction_error = mean(sum(@. ((x̄ - x)^2); dims=1))
  kl_divergence = mean(sum(@. ((μ^2 + exp(logσ²) - 1 - logσ²) / 2); dims=1))
  return reconstruction_error + kl_divergence
end
########################

#### Set up model #########
# image size is (64, 64)
Dhidden_efficientnet = 1792
Dlatent = 64

# Backbone:
# here we load the timm model and remove the classification part
using PyCall, PyNNTraining
torch, functorch = pyimport("torch"), pyimport("functorch.experimental")
backbone = load(models()["timm/efficientnetv2_rw_s"], pretrained=true);
backbone.classifier = torch.nn.Identity()  # 1792 dimensions
functorch.replace_all_batch_norm_modules_(backbone);

bridge = 
    Parallel(
        tuple,
        Dense(Dhidden_efficientnet, Dlatent), # μ
        Dense(Dhidden_efficientnet, Dlatent), # logσ²
    ) |> DEVICE

decoder = Chain(Dense(Dlatent, 16*16*32),
                xs -> reshape(xs, 16, 16, 32, :),
                Metalhead.basicblock(32, 128),
                PixelShuffle(2),
                Metalhead.basicblock(32, 128),
                PixelShuffle(2),
                Metalhead.basicblock(32, 3),
               ) |> DEVICE

model = VAE(backbone, bridge, bridge, decoder);
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
    gs = gradient(learner.params) do
      intermediate   = learner.model.encoder(state.x)
      μ, logσ²       = learner.model.bridge(intermediate)
      state.z        = sample_latent(μ, logσ²)
      state.x̄        = learner.model.decoder(state.z)

      handle(FluxTraining.LossBegin())
      state.loss = learner.lossfn(state.x, state.x̄, μ, logσ²)

      handle(FluxTraining.BackwardBegin())
      return state.loss
    end
    handle(FluxTraining.BackwardEnd())
    Flux.Optimise.update!(learner.optimizer, learner.params, gs)
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
learner = Learner(model, ELBO, callbacks=[ToPyTorch()])
# This seems to segfault randomly after a while.
fitonecycle!(learner, 5, 1e-4; phases=(VAETrainingPhase() => dl,
                                       VAEValidationPhase() => dl_val))
#####################################################

# xs = makebatch(task, data, rand(1:numobs(data), 4)) |> DEVICE
# ypreds, _ = model(xs)
# showoutputbatch(BE, task, cpu(xs), cpu(ypreds))
