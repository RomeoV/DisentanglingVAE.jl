import Flux: Chain, Conv, MeanPool, GlobalMeanPool, flatten
import Functors: @functor
import Metalhead: convnextblock

struct ScalarGate
  λ :: AbstractArray{Float32}
end
ScalarGate() = ScalarGate([rand(Float32)]*0f0)
@functor ScalarGate
function (sgate::ScalarGate)(x)
    sgate.λ .* x
end
Flux.trainable(sgate::ScalarGate) = (sgate.λ, )

ResidualBlock(c) = Parallel(+,
                            Chain(leakyrelu,
                                  Conv((3, 3), c=>c, identity; pad=SamePad()),
                                  leakyrelu,
                                  Conv((3, 3), c=>c, identity; pad=SamePad()),
                                  ScalarGate()),
                            identity)

ResidualEncoder(latent_dim; sc=1) = Chain(
                      Conv((5, 5), 3=>32÷sc, leakyrelu; stride=1, pad=SamePad()),
                      convnextblock(32÷sc),
                      convnextblock(32÷sc),
                      Conv((1, 1), 32÷sc=>64÷sc, identity; stride=1),
                      MeanPool((2, 2)),
                      convnextblock(64÷sc),
                      convnextblock(64÷sc),
                      MeanPool((2, 2)),
                      convnextblock(64÷sc),
                      convnextblock(64÷sc),
                      Conv((1, 1), 64÷sc=>latent_dim÷sc, identity; stride=1),
                      GlobalMeanPool(),
                      Flux.flatten)

"See ON THE TRANSFER OF DISENTANGLED REPRESENTATIONS IN REALISTIC SETTINGS, Appendix A
 https://arxiv.org/pdf/2010.14407.pdf"
ResidualDecoder(latent_dim; sc=1) = Chain(
      Dense(latent_dim, 256÷sc, leakyrelu),
      Dense(256÷sc, 1024÷sc, identity),  # identity here because we have relu in ResidualBlock
      x->reshape(x, 4, 4, 64÷sc, :),  # 4x4
      convnextblock(64÷sc),
      convnextblock(64÷sc),

      Upsample(2, :bilinear),  # 8x8
      convnextblock(64÷sc),
      convnextblock(64÷sc),
      Conv((1, 1), 64÷sc=>32÷sc, identity; stride=1),

      Upsample(2, :bilinear),  # 16x16
      convnextblock(32÷sc),
      convnextblock(32÷sc),

      Upsample(2, :bilinear),  # 32x32
      x->leakyrelu.(x),
      Conv((5, 5), 32÷sc=>3, identity; stride=1, pad=SamePad()),
      # xs -> xs[3:30, 3:30, :, :]  # 28x28
)
