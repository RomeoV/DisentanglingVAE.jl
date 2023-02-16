using FastAI, FastVision
import FastVision: ImageTensor
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
