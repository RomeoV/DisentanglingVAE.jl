module VAECallbacks
include("callbacks.jl")
include("eval_linear_model.jl")
export VisualizationCallback,
       LinearModelCallback,
       CSVLoggerBackend,
       ExpDirPrinterCallback

end
