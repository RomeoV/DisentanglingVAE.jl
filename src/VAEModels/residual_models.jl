import Flux: Chain, Dense, Conv, MeanPool, GlobalMeanPool, SamePad, flatten, leakyrelu
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
Flux.trainable(sgate::ScalarGate) = (λ=sgate.λ,)

ResidualBlock(c) = Parallel(+,
                            Chain(leakyrelu,
                                  Conv((3, 3), c=>c, identity; pad=SamePad()),
                                  leakyrelu,
                                  Conv((3, 3), c=>c, identity; pad=SamePad()),
                                  ScalarGate()),
                            identity)

ResidualEncoder(latent_dim; sc=1, Block_t=ResidualBlock) = Chain(
                      Conv((5, 5), 3=>32÷sc, leakyrelu; stride=1),
                      Block_t(32÷sc),
                      Block_t(32÷sc),
                      Conv((1, 1), 32÷sc=>64÷sc, identity; stride=2),
                      # MeanPool((2, 2)),
                      Block_t(64÷sc),
                      Block_t(64÷sc),
                      Conv((1, 1), 64÷sc=>64÷sc, identity; stride=2),
                      # MeanPool((2, 2)),
                      Block_t(64÷sc),
                      Block_t(64÷sc),
                      Conv((1, 1), 64÷sc=>latent_dim÷sc, identity; stride=2),
                      GlobalMeanPool(),
                      Flux.flatten)


"See ON THE TRANSFER OF DISENTANGLED REPRESENTATIONS IN REALISTIC SETTINGS, Appendix A
 https://arxiv.org/pdf/2010.14407.pdf"
ResidualDecoder(latent_dim; sc=1, Block_t=ResidualBlock) = Chain(
      Dense(latent_dim, 256÷sc, leakyrelu),
      Dense(256÷sc, 1024÷sc, identity),  # identity here because we have relu in ResidualBlockt
      x->reshape(x, 4, 4, 64÷sc, :),  # 4x4
      Block_t(64÷sc),
      Block_t(64÷sc),
      Conv((1, 1), 64÷sc=>32÷sc, identity; stride=1),

      Upsample(2, :bilinear),  # 8x8
      Block_t(32÷sc),
      Block_t(32÷sc),
      Conv((1, 1), 32÷sc=>16÷sc, identity; stride=1),

      Upsample(2, :bilinear),  # 16x16
      Block_t(16÷sc),
      Block_t(16÷sc),

      Upsample(2, :bilinear),  # 32x32
      x->leakyrelu.(x),
      # Conv((1, 1), 16÷sc=>3, identity; stride=1),
      Conv((5, 5), 16÷sc=>3, identity; stride=1, pad=SamePad()),
      # xs -> xs[3:30, 3:30, :, :]  # 28x28
)
