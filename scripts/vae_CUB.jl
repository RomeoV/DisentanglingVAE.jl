using FastAI
using FastAI.FluxTraining
using FastAI.Flux
using FastVision
import FastVision: Gray, SVector, RGB
using FastTimm
import ChainRulesCore: @ignore_derivatives
import Metalhead
import FastAI.Flux.MLUtils._default_executor
using FastAI.MLUtils.Transducers: ThreadedEx
_default_executor() = ThreadedEx()
# FastAI.SHOW_BACKEND[] = ShowText();
DEVICE = gpu
# some memory utilities
# torch.cuda.empty_cache()
# import FastAI.Flux.CUDA
# CUDA.memory_status()

# using FastMakie
# import GLMakie

# using ImageInTerminal
# notice that we're using 'datasets' not 'datarecipes'
BE = ShowText();
# path = load(datasets()["CUB_200_2011"])
path = load(datasets()["mnist_tiny"])
data = Datasets.loadfolderdata(
    path,
    filterfn = FastVision.isimagefile,
    loadfn = loadfile,
)
# data, blocks = load(datarecipes()["CUB_200_2011"])
showblock(BE, Image{2}(), getobs(data, 3))



# Note: logsigma and mu are probably modeled as block "continuous" with some dimension


using FastAI: encodedblock, encodedblockfilled, decodedblockfilled

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

x = encodesample(task, Training(), getobs(data, 1));
showencodedsample(BE, task, x)

td = taskdataset(data, task, Training())
BATCHSIZE=2
dl, dl_val = taskdataloaders(data, task, BATCHSIZE, pctgval=0.1;
                             buffer=true, partial=false);


## Define model
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

using Random: randn!
using Statistics: mean

sample_latent(μ::AbstractArray{T}, logσ²::AbstractArray{T}) where {T} =
       μ .+ exp.(logσ²./2) .* randn!(similar(logσ²))

function ELBO(x, x̄, μ, logσ²)
  reconstruction_error = mean(sum(@. ((x̄ - x)^2); dims=1))
  kl_divergence = mean(sum(@. ((μ^2 + exp(logσ²) - 1 - logσ²) / 2); dims=1))
  return reconstruction_error + kl_divergence
end

SIZE = (64, 64, 3)
# Din = prod(SIZE)
# Dhidden = 512
Dlatent = 64

# encoder =
#     Chain(
#         Flux.flatten,
#         Dense(Din, Dhidden, relu), # backbone
#         Parallel(
#             tuple,
#             Dense(Dhidden, Dlatent), # μ
#             Dense(Dhidden, Dlatent), # logσ²
#         ),
#     ) |> gpu



decoder = Chain(Dense(Dlatent, 16*16*32),
                xs -> reshape(xs, 16, 16, 32, :),
                Metalhead.basicblock(32, 128),
                PixelShuffle(2),
                Metalhead.basicblock(32, 128),
                PixelShuffle(2),
                Metalhead.basicblock(32, 3),
               ) |> DEVICE

# Run:
# ENV["PYTHON"] = "/home/romeo/.cache/pypoetry/virtualenvs/intervention-experiments-poetry-CVVio-gT-py3.10/bin/python"
# ] build PyCall
# restart julia
using PyCall, PyNNTraining
torch = pyimport("torch")
fct = pyimport("functorch.experimental")

# timm, torch = pyimport("timm"), pyimport("torch")
# backbone = timm.create_model("resnet18", pretrained=true);
# backbone.fc = torch.nn.Identity();

effnet = load(models()["timm/efficientnetv2_rw_s"], pretrained=true);
effnet.classifier = torch.nn.Identity()  # 1792 dimensions
backbone = effnet;
# backbone.requires_grad_(false)
fct.replace_all_batch_norm_modules_(backbone);

Dhidden_effnet = 1792
bridge = 
    Parallel(
        tuple,
        Dense(Dhidden_effnet, Dlatent), # μ
        Dense(Dhidden_effnet, Dlatent), # logσ²
    ) |> DEVICE

model = VAE(backbone, bridge, decoder);
params = Flux.params(bridge, decoder);

struct VAETrainingPhase <: FluxTraining.AbstractTrainingPhase end
struct VAEValidationPhase <: FluxTraining.AbstractValidationPhase end

function FluxTraining.step!(learner, phase::VAETrainingPhase, batch)
  FluxTraining.runstep(learner, phase, (x = batch,)) do handle, state
    # we "freeze" these weights by not including the parameters
    # in the optimizer
    gs = gradient(learner.params) do
      # intermediate   = @ignore_derivatives learner.model.encoder(state.x)
      intermediate   = learner.model.encoder(state.x)
      # we could also move this in front of the "gradient" call
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



function FluxTraining.on(
    ::FluxTraining.StepBegin,
    ::Union{VAETrainingPhase, VAEValidationPhase},
    cb::ToDevice,
    learner,
  )
  learner.step.x = cb.movedatafn(learner.step.x)
end

# learner = Learner(model, ELBO, callbacks=[ToGPU()])
learner = Learner(model, ELBO, callbacks=[ToPyTorch()])

# dataiter = collect(dl)
fitonecycle!(learner, 5, 0.01; phases=(VAETrainingPhase() => dl,
                                       VAEValidationPhase() => dl_val))

xs = makebatch(task, data, rand(1:numobs(data), 4)) |> DEVICE
ypreds, _ = model(xs)
showoutputbatch(BE, task, cpu(xs), cpu(ypreds))
