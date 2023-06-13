import Flux: Dense, Conv, Chain, Parallel, Upsample, leakyrelu

resnet_backbone() = let backbone = Metalhead.ResNet(18; pretrain=false)
   Chain(backbone.layers[1], Chain(backbone.layers[2].layers[1:2]..., identity))
end
# convnext_backbone() = Metalhead.ConvNeXt(:tiny; nclasses=backbone_dim)
convnext_backbone() = let backbone = Metalhead.ConvNeXt(:tiny)
    # 768 output dimension by default
    Chain(backbone.layers[1], Chain(backbone.layers[2].layers[[1, 2]]...))
end
backbone() = convnext_backbone()

bridge(backbone_dim, latent_dim; layernorm=true) = Chain(
          Dense(backbone_dim, 128, leakyrelu),
          (layernorm ? LayerNorm(128) : identity),
          backbone_head(128=>latent_dim),
        )

backbone_head(in_out::Pair{Int, Int}) = let (in_dim, out_dim) = in_out
    Parallel(
        tuple,
        Dense(1//10*Flux.glorot_uniform(out_dim, in_dim),
              zeros(Float32, out_dim)),  # mu
        # Special initialization, see https://arxiv.org/pdf/2010.14407.pdf, Table 2 (Appendix)
        Dense(1//10*Flux.glorot_uniform(out_dim, in_dim),
              -1*ones(Float32, out_dim)),  # logvar
        Dense(in_dim=>out_dim),  # escape
      )
end


decoder() = Chain(Dense(latent_dim, 4*4*16, leakyrelu),
                Dense(4*4*16, 4*4*64, identity),
                xs -> reshape(xs, 4, 4, 64, :),
                ResidualBlock(64),
                ResidualBlock(64),
                Upsample(2),
                ResidualBlock(64),
                ResidualBlock(64),
                Conv((1, 1), 64=>32, identity),
                ResidualBlock(32),
                ResidualBlock(32),
                Upsample(2),
                Conv((1, 1), 32=>16, identity),
                Upsample(2),
                leakyrelu,
                Conv((5, 5), 16=>3, identity; pad=SamePad(), stride=1),
                xs -> xs[3:30, 3:30, :, :])
