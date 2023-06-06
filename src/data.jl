module LineData
import Distributions: Normal, Distribution, TaskLocalRNG
import Random: seed!
import FastAI: datarecipes, load
import FastVision
using ReTest
import ChainRulesCore: @ignore_derivatives

include("line_utils.jl")

export make_data_sample

make_data_sample(i::Int; kwargs...) = make_data_sample(Normal(0., 1/2), i;
                                                       kwargs...)
make_data_sample(xs::UnitRange; kwargs...) = make_data_sample.(xs; kwargs...)

const ((cifar10_x, cifar10_y), cifar10_blocks) = load(datarecipes()["cifar10"])
airplane_img(i) = cifar10_x[(i % 15_000)+1]
# make_data_sample_(i::Int) = make_data_sample(Normal(0, 0.5), i;
#                                              x0_fn = i->0.1*airplane_img(i))  # 15_000 airplanes
make_data_sample_(i::Int) = let sample = make_data_sample(Normal(0, 0.5), i)
  (sample, sample[1:4])
end


# Note: We need a bit of noise, otherwise our NN runs into numerical issues.
rand_x0(_) = 0.1f0*rand(RGB{Float32}, 32, 32)
function make_data_sample(D::Distribution, i::Int;
                          x0_fn=rand_x0)
  # the ks are sampled truely randomly, i.e. with a device that is not seeded
  # each concept has a chance of being forced to be "the same"
  rng = TaskLocalRNG()
  seed!(rng, i)
  k = rand(rng, 1:6)
  ks = ones(Bool, 6); ks[k] = false

  v_lhs = rand(rng, D, 6)
  v_rhs = rand(rng, D, 6)
  v_rhs[ks] .= v_lhs[ks]
  v_lhs[k], v_rhs[k] = minmax(v_lhs[k], v_rhs[k])  # we sort such that lhs < rhs always at k
  if v_rhs[k] - v_lhs[k] < 0.5
      let v̄ = (v_rhs[k] + v_lhs[k])/2
          v_rhs[k] = v̄ + max((v_rhs[k] - v̄)*3, 0.25)
          v_lhs[k] = v̄ - max((v̄ - v_lhs[k])*3, 0.25)
      end
  end
  x0 :: Matrix{RGB{Float32}} = x0_fn(i) .|> RGB{Float32}

  img_lhs = copy(x0)
  draw!(img_lhs, v_lhs[1:2]..., RGB{Float32}(1.,0,0))
  draw!(img_lhs, v_lhs[3:4]..., RGB{Float32}(0,1.,0))
  draw!(img_lhs, v_lhs[5:6]..., RGB{Float32}(0,0,1.))

  img_rhs = copy(x0)
  draw!(img_rhs, v_rhs[1:2]..., RGB{Float32}(1.,0,0))
  draw!(img_rhs, v_rhs[3:4]..., RGB{Float32}(0,1.,0))
  draw!(img_rhs, v_rhs[5:6]..., RGB{Float32}(0,0,1.))

  @ignore_derivatives ((img_lhs, v_lhs, img_rhs, v_rhs, float.(ks)),
                       (img_lhs, v_lhs, img_rhs, v_rhs))
end

@testset "line data tests" begin
  make_data_sample(1)
  @test true
end

end
