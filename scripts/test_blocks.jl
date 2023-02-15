using FastAI, FastVision
import FastVision: RGB, ImageTensor
import FastAI.Flux.Losses: binarycrossentropy
import FastAI: showencodedsample, showsample
BE = ShowText();

task = begin
  BoundedImgTensor = Bounded{2, ImageTensor{2}}(ImageTensor{2}(3), (32, 32));
  sample = (Image{2}(), Continuous(6))
  x = BoundedImgTensor
  y = (BoundedImgTensor, Continuous(6))
  ŷ = (BoundedImgTensor, Continuous(6))
  encodedsample = (BoundedImgTensor, Continuous(6))
  enc = ( ProjectiveTransforms((32, 32)),

          ImagePreprocessing(means=FastVision.SVector(0., 0., 0.),
                             stds=FastVision.SVector(1., 1., 1.);
                             C = RGB{Float32},
                             buffered=false,
                            ),
         )
  BlockTask((; sample, x, y, ŷ, encodedsample), enc)
end

FastAI.testencoding(enc, (Image{2}(), Continuous(6)))

sample_fn(_) = begin
  # sample = (rand(RGB, 64, 64), rand(6))
  sample = FastAI.mocksample(task)
  sample
end
data = mapobs(sample_fn, 1:64);
x = encodesample(task, Training(), getobs(data, 1))
showencodedsample(BE, task, x)

model = identity

BATCHSIZE=8;
dl, _ = taskdataloaders(data, task, BATCHSIZE, pctgval = 0.1);
learner = Learner(identity, (y_pred, y)->binarycrossentropy(y_pred[2], y[2]),
                  data=(dl, dl),)
fitonecycle!(learner, 1)

xs = makebatch(task, data, rand(1:numobs(data), 4))
tpl = model(xs);
showoutputbatch(BE, task, xs, tpl)
# showoutputbatch(BE, task, cpu(xs), cpu(ypreds) .|> sigmoid)


