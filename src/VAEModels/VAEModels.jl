module VAEModels
import MLUtils
import Flux
import Flux.Zygote
import Metalhead
import ReTest: @test, @testset

include("flux_patches.jl")
include("residual_models.jl")
include("layers.jl")
include("model.jl")

export VAE
export ResidualBlock, ResidualEncoder, ResidualDecoder,
       backbone_head

import PrecompileTools
PrecompileTools.@compile_workload begin
    # Precompiling for gpu currently has problems, see https://github.com/JuliaGPU/CUDA.jl/issues/1870
    # for device in [Flux.cpu, Flux.gpu]
    for device in [Flux.cpu]
        model = VAE() |> device
        x = rand(Float32, 32, 32, 3, 7) |> device
        model(x)
        gs = Zygote.gradient(Flux.params(model)) do
            model(x)[1] |> sum
        end
    end
end
end
