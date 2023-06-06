import FastAI
import FastAI: Block, ShowText

"""
    Continuous{T} is a modification of FastAI.Continuous but with variable type.
    We need this to get Float32 mockblock results.
"""
struct Continuous{T<:Number} <: Block
    size::Int
end
Continuous32 = Continuous{Float32}

function FastAI.checkblock(block::Continuous{T}, x) where T
    block.size == length(x) && eltype(x) <: Number
end

FastAI.mockblock(block::Continuous{T}) where T = rand(T, block.size)

function FastAI.blocklossfn(outblock::Continuous{T}, yblock::Continuous{T}) where T
    outblock.size == yblock.size || error("Sizes of $outblock and $yblock differ!")
    return Flux.Losses.mse
end

# we patch to show vectors as columns
function FastAI.showblock!(io, ::ShowText, block::Continuous{T}, obs::AbstractArray) where T
    print(io, join(round.(obs, sigdigits=3), '\n'))
end
