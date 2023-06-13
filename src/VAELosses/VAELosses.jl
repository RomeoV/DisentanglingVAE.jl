module VAELosses
import ReTest: @testset, @test
include("losses.jl")
include("loss.jl")

export reg_l1, reg_l2, gaussian_nll, directionality_loss, cov_loss, kl_divergence
export VAELoss, LossSchedule, LossConfig, LinearWarmupSchedule, LossParam

end
