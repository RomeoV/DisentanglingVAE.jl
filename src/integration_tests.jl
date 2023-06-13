@testset "model step" for device in [Flux.cpu, Flux.gpu]
  import ..VAEModels: VAE
  model = VAE()
  learner = Learner(model, VAELoss{Float64}();
                    optimizer=Optimisers.Adam(),
                    callbacks=[FluxTraining.ToDevice(device, device)])

  task = DisentanglingVAETask()
  x_, y_ = FastAI.mocksample(task)
  # x = FastAI.encodeinput(task, FastAI.Training(), x_)
  y = FastAI.encodetarget(task, FastAI.Training(), y_)
  x = y[[1, 3]]
  xs = batch([x, x])
  ys = batch([y, y])

  FluxTraining.step!(learner, VAETrainingPhase(), (xs, ys))
  @test true
end
