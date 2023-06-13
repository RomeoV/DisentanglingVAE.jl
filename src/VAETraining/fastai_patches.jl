import FastAI
import FastAI: Block, ShowText

"""
    Continuous_{T} is a modification of FastAI.Continuous but with variable type.
    We need this to get Float32 mockblock results.
"""
struct Continuous_{T<:Number} <: Block
    size::Int
end
Continuous32 = Continuous_{Float32}

function FastAI.checkblock(block::Continuous_{T}, x) where T
    block.size == length(x) && eltype(x) <: Number
end

FastAI.mockblock(block::Continuous_{T}) where T = rand(T, block.size)

function FastAI.blocklossfn(outblock::Continuous_{T}, yblock::Continuous_{T}) where T
    outblock.size == yblock.size || error("Sizes of $outblock and $yblock differ!")
    return Flux.Losses.mse
end

# we patch to show vectors as columns
function FastAI.showblock!(io, ::ShowText, block::Continuous_{T}, obs::AbstractArray) where T
    print(io, join(round.(obs, sigdigits=3), '\n'))
end
