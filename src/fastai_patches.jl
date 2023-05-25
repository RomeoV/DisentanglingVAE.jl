import FastAI
import FastAI: ShowBackend, AbstractBlockTask
import FastAI: showblockinterpretable, getencodings, getblocks
import FastAI: Continuous, ShowText
import Printf: @sprintf

# we patch to remove the dictionary structure
# which causes a bug down the line I think
function FastAI.showencodedsample(backend::ShowBackend, task::AbstractBlockTask, encsample::Tuple)
    showblockinterpretable(backend,
                           getencodings(task),
                           getblocks(task).encodedsample,
                           encsample)
end

# we patch to show vectors as columns
function FastAI.showblock!(io, ::ShowText, block::Continuous, obs::AbstractArray)
    print(io, join(round.(obs, sigdigits=3), '\n'))
end

FastAI.mockblock(block::Continuous) = rand(Float32, block.size)

