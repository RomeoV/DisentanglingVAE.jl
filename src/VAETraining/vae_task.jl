import FastAI
import FastAI: SupervisedTask
import FastVision
import FastVision: RGB, Image, ProjectiveTransforms, ImagePreprocessing
import Random: seed!, RandomDevice, TaskLocalRNG

function DisentanglingVAETask(; sz=(32, 32))
    tpl_in   = (Image{2}(), Image{2}())
    tpl_pred = (Image{2}(), Continuous32(6), Continuous32(6),   # LHS: x̄, μ, logσ²,
                Image{2}(), Continuous32(6), Continuous32(6))   # RHS: x̄, μ, logσ²
    tpl_out  = (Image{2}(), Continuous32(6),                    # LHS: x, v
                Image{2}(), Continuous32(6),                    # RHS: x, v
                Continuous32(6))                                # ks content

    encodings = (ProjectiveTransforms(sz),
                    ImagePreprocessing(means=FastVision.SVector(0., 0., 0.),
                                       stds =FastVision.SVector(1., 1., 1.)))

    SupervisedTask((tpl_in, tpl_out), encodings;
                   ŷblock=FastAI.encodedblockfilled(encodings, tpl_pred))  # we have to encode this ourselves...
end

function EncoderTask(sz)
  SupervisedTask((Image{2}(), Continuous32(6)),
                 (ProjectiveTransforms((32, 32)),
                  ImagePreprocessing(means = FastVision.SVector(0., 0., 0.),
                                     stds  = FastVision.SVector(1., 1., 1.);
                                     C     = RGB{Float32},
                                     buffered=false,),))
end

function DecoderTask(sz)
  SupervisedTask((Continuous32(6), Image{2}()),
                 # (ProjectiveTransforms((32, 32)),))
                  (ImagePreprocessing(means = FastVision.SVector(0., 0., 0.),
                                      stds  = FastVision.SVector(1., 1., 1.);
                                      C     = RGB{Float32},
                                      buffered=false,),))
end