module VAEData
using ReTest
include("line_utils.jl")
include("data.jl")
export make_data_sample

import PrecompileTools
PrecompileTools.@compile_workload begin
    make_data_sample(1)
end

end
