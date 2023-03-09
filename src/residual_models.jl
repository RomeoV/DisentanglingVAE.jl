import Flux: Chain, Conv, MeanPool, flatten
import Functors: @functor

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

ResidualEncoder(; sc=1) = Chain(
                      Conv((5, 5), 3=>32, relu; stride=1, pad=SamePad()),
                      Conv((1, 1), 32=>64, relu; stride=2),
                      Conv((1, 1), 64=>64, relu; stride=2),
                      Conv((1, 1), 64=>128, relu; stride=2),
                      x->reshape(x, 4*4*128, :),
                     )

ResidualEncoderWithHead(latent_dim; sc=1) = Chain(
                Chain(
                      Conv((5, 5), 3=>32÷sc, leakyrelu; stride=1, pad=SamePad()),
                      ResidualBlock(32÷sc),
                      ResidualBlock(32÷sc),
                      Conv((1, 1), 32÷sc=>64÷sc, identity; stride=1),
                      MeanPool((2, 2)),
                      ResidualBlock(64÷sc),
                      ResidualBlock(64÷sc),
                      MeanPool((2, 2)),
                      ResidualBlock(64÷sc),
                      ResidualBlock(64÷sc),
                      Conv((1, 1), 64÷sc=>128÷sc, identity; stride=1),
                      MeanPool((2, 2)),
                      Flux.flatten,
                      Dense(4*4*128÷sc, 128÷sc, leakyrelu),
                      LayerNorm(128÷sc),
                     ),
                Parallel(tuple,
                    # see https://arxiv.org/pdf/2010.14407.pdf, Table 2 (Appendix)
                    Dense(0.1*Flux.glorot_uniform(latent_dim, 128÷sc), -1*ones(latent_dim)),
                    Dense(0.1*Flux.glorot_uniform(latent_dim, 128÷sc), -1*ones(latent_dim))
                ))


"See ON THE TRANSFER OF DISENTANGLED REPRESENTATIONS IN REALISTIC SETTINGS, Appendix A
 https://arxiv.org/pdf/2010.14407.pdf"
ResidualDecoder(latent_dim; sc=1) = Chain(
      Dense(latent_dim, 256÷sc, leakyrelu),
      Dense(256÷sc, 1024÷sc, identity),  # identity here because we have relu in ResidualBlock
      x->reshape(x, 4, 4, 64÷sc, :),  # 4x4
      ResidualBlock(64÷sc),
      ResidualBlock(64÷sc),

      Upsample(2, :bilinear),  # 8x8
      ResidualBlock(64÷sc),
      ResidualBlock(64÷sc),
      Conv((1, 1), 64÷sc=>32÷sc, identity; stride=1),

      Upsample(2, :bilinear),  # 16x16
      ResidualBlock(32÷sc),
      ResidualBlock(32÷sc),

      Upsample(2, :bilinear),  # 32x32
      x->leakyrelu.(x),
      Conv((5, 5), 32÷sc=>3, identity; stride=1, pad=SamePad()),
      # xs -> xs[3:30, 3:30, :, :]  # 28x28
)

ResidualBlockOLD(c) = Parallel(+,
                            Chain(leakyrelu,
                                  Conv((3, 3), c=>c, identity; pad=SamePad()),
                                  leakyrelu,
                                  Conv((3, 3), c=>c, identity; pad=SamePad())),
                            identity)
ResidualDecoderOLD(latent_dim; sc=1) = Chain(
      Dense(latent_dim, 256÷sc, leakyrelu),
      Dense(256÷sc, 1024÷sc, identity),  # identity here because we have relu in ResidualBlock
      x->reshape(x, 4, 4, 64÷sc, :),  # 4x4
      ResidualBlockOLD(64÷sc),
      ResidualBlockOLD(64÷sc),

      Upsample(2, :bilinear),  # 8x8
      ResidualBlockOLD(64÷sc),
      ResidualBlockOLD(64÷sc),
      Conv((1, 1), 64÷sc=>32÷sc, identity; stride=1),

      Upsample(2, :bilinear),  # 16x16
      ResidualBlockOLD(32÷sc),
      ResidualBlockOLD(32÷sc),

      Upsample(2, :bilinear),  # 32x32
      x->leakyrelu.(x),
      Conv((5, 5), 32÷sc=>3, identity; stride=1, pad=SamePad()),
      # xs -> xs[3:30, 3:30, :, :]  # 28x28
)
