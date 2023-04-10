import LinearAlgebra: diag

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
