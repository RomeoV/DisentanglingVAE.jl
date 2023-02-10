module DisentanglingVAE
using FastAI
using FastAI.Flux
using Metalhead
include("line_utils.jl")
include("model.jl")
include("task.jl")

export InputBlockTuple, InputBlockTuple_, OutputBlockTuple, OutputBlockTuple_, DisentanglingVAETask

greet() = print("Hello World!")

end # module DisentanglingVAE
