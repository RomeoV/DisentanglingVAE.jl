import FastVision: RGB
import ImageCore.ColorTypes: mapc

make_X(dims) = ((x, y) for x in LinRange(0, 1, dims[1]),
                           y in LinRange(0, 1, dims[2]));

# we assume Δy and α are from N(0,1), and scale them accordingly
function draw!(img::Matrix{RGB{Float32}}, Δy::R, α::R, C::RGB{Float32}) where R<:Real
  p1 = (0.5+Δy/4, 0.5)
  p2 = (0.5+Δy/4 + cos(α*pi/4), 0.5+sin(α*pi/4))
  f(t) = 2 ./ exp.(800*dist_point_to_line(p1, p2, t)^2)
  X = make_X(size(img))
  Z = f.(X)
  img_ = mapc.(x->clamp(x, 0., 1.),
               float(img) + Z.*float(C))
  img .= convert.(RGB{Float32}, img_)
end

"Ortogonal distance from x_0 to line through x_1->x_2"
dist_point_to_line((x1, y1)::Tuple, (x2, y2)::Tuple, (x0, y0)::Tuple) =
         abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / sqrt((x2-x1)^2 + (y2-y1)^2)
