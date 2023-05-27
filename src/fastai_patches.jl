import FastAI
import FastAI: Continuous, ShowText

# we patch to show vectors as columns
function FastAI.showblock!(io, ::ShowText, block::Continuous, obs::AbstractArray)
    print(io, join(round.(obs, sigdigits=3), '\n'))
end

# function FastAI.mockblock(block::Continuous)
#   rand(Float32, block.size)
# end
