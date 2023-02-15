ResidualEncoder(latent_dim; sc=1) = Encoder(
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
                      Dense(3*3*128÷sc, 128÷sc, leakyrelu),
                      LayerNorm(128÷sc),
                     ),
                # see https://arxiv.org/pdf/2010.14407.pdf, Table 2 (Appendix)
                Chain(Dense(0.1*Flux.glorot_uniform(latent_dim, 128÷sc), -1*ones(latent_dim)),),
                Chain(Dense(0.1*Flux.glorot_uniform(latent_dim, 128÷sc), -1*ones(latent_dim)),),  # var (always positive)
               )


"See ON THE TRANSFER OF DISENTANGLED REPRESENTATIONS IN REALISTIC SETTINGS, Appendix A
 https://arxiv.org/pdf/2010.14407.pdf"
ResidualDecoder(latent_dim; sc=1) = Chain(
      Dense(latent_dim, 256÷sc, leakyrelu),
      Dense(256÷sc, 1024÷sc, identity),  # identity here because we have relu in ResidualBlock
      x->reshape(x, 4, 4, 64÷sc, :),
      ResidualBlock(64÷sc),
      ResidualBlock(64÷sc),

      Upsample(2, :bilinear),
      ResidualBlock(64÷sc),
      ResidualBlock(64÷sc),
      Conv((1, 1), 64÷sc=>32÷sc, identity; stride=1),

      Upsample(2, :bilinear),
      ResidualBlock(32÷sc),
      ResidualBlock(32÷sc),

      Upsample(2, :bilinear),
      x->leakyrelu.(x),
      Conv((5, 5), 32÷sc=>3, identity; stride=1),
)
