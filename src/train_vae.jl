using StatsBase: sample, mean
using FastAI: ObsView, mapobs, taskdataloaders
using Random: seed!

function make_data_sample(i)
  seed!(i)
  ks = rand(Bool, 6)
  std = 1/sqrt(2)
  v_lhs, v_rhs = randn(Float32, 6, 1)*std, randn(Float32, 6, 1)*std
  v_lhs[ks] .= v_rhs[ks]
  (; v_lhs, v_rhs, ks)
end
data = mapobs(make_data_sample, ObsView(1:2048))

task = DisentanglingVAETask(InputBlockTuple_(6),
                            LineTupleEncoding())

BATCHSIZE=8
dl, dl_val = taskdataloaders(data, task, BATCHSIZE, pctgval=0.1;
                             buffer=false, partial=false,
                             parallel=false
                            );  # for debug

DEVICE = gpu
model = VAE(backbone(), bridge(), decoder(), DEVICE);
params = Flux.params(model.bridge, model.decoder);

#### Try to run the training. #######################
opt = Flux.Optimiser(ClipNorm(1), Adam())
# opt = Flux.Optimiser(Adam())
learner = Learner(model, ELBO;
                  optimizer=opt,
                  data=(dl, dl_val),
                  callbacks=[ToGPU(), ProgressPrinter()])

# test one input
@ignore_derivatives model(getbatch(learner)[1] |> gpu)
fitonecycle!(learner, 5, 1e-4; phases=(VAETrainingPhase() => dl,
                                        VAEValidationPhase() => dl_val))
#####################################################

xs = makebatch(task, data, rand(1:numobs(data), 4)) |> DEVICE;
ypreds, _ = model(xs);
showoutputbatch(BE, task, cpu(xs), cpu(ypreds) .|> sigmoid)
