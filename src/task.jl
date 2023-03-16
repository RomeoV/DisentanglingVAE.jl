import DisentanglingVAE
using FastAI, FastVision
import FastAI: Continuous
import FastVision: RGB
import FastVision: ImageTensor
using Random: seed!, RandomDevice
import Distributions: Distribution, Normal

function DisentanglingVAETask()
  BoundedImgTensor(dim) = Bounded{2, ImageTensor{2}}(ImageTensor{2}(3), (dim, dim));
  sample = (Image{2}(), Continuous(6), Image{2}(), Continuous(6), Continuous(6))
  x = BoundedImgTensor(32)
  y = (BoundedImgTensor(32), Continuous(6),
       BoundedImgTensor(32), Continuous(6))
  ŷ = (BoundedImgTensor(32), Continuous(6),
       BoundedImgTensor(32), Continuous(6))
  encodedsample = (BoundedImgTensor(32), Continuous(6),
                   BoundedImgTensor(32), Continuous(6),
                   Continuous(6))
  enc = ( ProjectiveTransforms((32, 32)),
          ImagePreprocessing(means=FastVision.SVector(0., 0., 0.),
                             stds=FastVision.SVector(1., 1., 1.);
                             C = RGB{Float32},
                             buffered=false,
                            ),
         )
  BlockTask((; sample, x, y, ŷ, encodedsample), enc)
end

make_data_sample(i::Int) = make_data_sample(Normal, i)

rand_x0(_) = 0.1f0*rand(RGB{Float32}, 32, 32)
function make_data_sample(DT::Type{<:Distribution}, i::Int; Dargs=(0.f0, 0.5f0), x0_fn=rand_x0)
  # the ks are sampled truely randomly, i.e. with a device that is not seeded
  # each concept has a chance of being forced to be "the same"
  k = rand(RandomDevice(), 1:6)
  ks = ones(Bool, 6); ks[k] = false
  seed!(i)
  D = DT(Dargs...)
  v_lhs = rand(D, 6)
  v_rhs = rand(D, 6)
  v_rhs[ks] .= v_lhs[ks]
  x0 = x0_fn(i) .|> RGB{Float32}

  img_lhs = copy(x0)
  DisentanglingVAE.draw!(img_lhs, v_lhs[1:2]..., RGB{Float32}(1.,0,0))
  DisentanglingVAE.draw!(img_lhs, v_lhs[3:4]..., RGB{Float32}(0,1.,0))
  DisentanglingVAE.draw!(img_lhs, v_lhs[5:6]..., RGB{Float32}(0,0,1.))

  img_rhs = copy(x0)
  DisentanglingVAE.draw!(img_rhs, v_rhs[1:2]..., RGB{Float32}(1.,0,0))
  DisentanglingVAE.draw!(img_rhs, v_rhs[3:4]..., RGB{Float32}(0,1.,0))
  DisentanglingVAE.draw!(img_rhs, v_rhs[5:6]..., RGB{Float32}(0,0,1.))

  (img_lhs, v_lhs, img_rhs, v_rhs, ks)
end
