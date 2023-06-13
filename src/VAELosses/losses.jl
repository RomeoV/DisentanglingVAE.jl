import LinearAlgebra: diag
import Flux: leakyrelu
import StatsBase: mean

kl_divergence(μ, logσ²; agg=mean) = sum(@. ((μ^2 + exp(logσ²) - 1 - logσ²) / 2); dims=1) |> agg

# The StatsBase.cov implementation can not be differentiated somehow...
function cov_(z_batch::AbstractMatrix{<:Real}; corrected=true) :: AbstractMatrix{eltype(z_batch)}
    z̄ = mean(z_batch; dims=2)
    ((z_batch.-z̄) * (z_batch .- z̄)') ./ (size(z_batch, 2) - corrected)
end

function cov_loss(z_batch::AbstractMatrix{<:Real}) :: eltype(z_batch)
    d = size(z_batch, 1)
    C = cov_(z_batch)
    (sum(C.^2) - sum(diag(C).^2)) / d
end

"We want to encode rhs > lhs"
function directionality_loss(μ_lhs, μ_rhs)
    sum(leakyrelu(μ_lhs - μ_rhs))
end

function gaussian_nll((μ_pred, logvar_pred)::Tuple, y_target; eps=1e-6)
    @assert size(μ_pred) == size(y_target)
    mean(
        @. 0.5 * ( max(logvar_pred, log(eps)) + (μ_pred - y_target)^2 / max(exp(logvar_pred), eps) )
    )
end

reg_l1(params) = sum(x->sum(abs.(x)), params)
reg_l2(params) = sum(x->sum(x.^2), params)
