import FastVision: RGB
import ImageCore.ColorTypes: mapc

make_X(dims) = ((x, y) for x in LinRange(0, 1, dims[1]),
                           y in LinRange(0, 1, dims[2]));

"Ortogonal distance from x_0 to line through x_1->x_2"
dist_point_to_line((x1, y1)::Tuple, (x2, y2)::Tuple, (x0, y0)::Tuple) = 
         abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / sqrt((x2-x1)^2 + (y2-y1)^2)

function draw!(img::Array{T, 3}, Δy::R, α::R, c::Vector{T}) where T<:Real where R<:Real
  p1 = (0.5+Δy, 0.5)
  p2 = (0.5+Δy+cos(α*pi/2), 0.5+sin(α*pi/2))
  f(t) = 2 ./ exp.(800*dist_point_to_line(p1, p2, t)^2)
  X = make_X(size(img)[2:3])
  Z = f.(X)

  img .+= reshape(Z, 1, size(Z)...) .* reshape(c, :, 1, 1)
  clamp!(img, 0., 1.)
end
function draw!(img::Matrix{RGB{Float32}}, Δy::R, α::R, C::RGB{Float32}) where R<:Real
  p1 = (0.5+Δy, 0.5)
  p2 = (0.5+Δy+cos(α*pi/2), 0.5+sin(α*pi/2))
  f(t) = 2 ./ exp.(800*dist_point_to_line(p1, p2, t)^2)
  X = make_X(size(img))
  Z = f.(X)
  img_ = mapc.(x->clamp(x, 0., 1.),
               float(img) + Z.*float(C))
  img .= convert.(RGB{Float32}, img_)
end
function draw(dims::Tuple{Int, Int}, Δy::R, α::R, C::RGB{Float32}) where R<:Real
  img = zeros(RGB{Float32}, dims...)
  p1 = (0.5+Δy, 0.5)
  p2 = (0.5+Δy+cos(α*pi/2), 0.5+sin(α*pi/2))
  f(t) = 2 ./ exp.(800*dist_point_to_line(p1, p2, t)^2)
  X = make_X(size(img))
  Z = f.(X)
  img_ = mapc.(x->clamp(x, 0., 1.),
               float(img) + Z.*float(C))
  img .= convert.(RGB{Float32}, img_)
end

function draw_labels_on_image_two_ways!((img_lhs, img_rhs)::Tuple,
                                        (labels_lhs, labels_rhs)::Tuple)
  draw!(img_lhs, labels_lhs[1], labels_lhs[2], Float32[1., 0., 0.])
  draw!(img_lhs, labels_lhs[3], labels_lhs[4], Float32[0., 1., 0.])
  draw!(img_lhs, labels_lhs[5], labels_lhs[6], Float32[0., 0., 1.])

  draw!(img_rhs, labels_rhs[1], labels_rhs[2], Float32[1., 0., 0.])
  draw!(img_rhs, labels_rhs[3], labels_rhs[4], Float32[0., 1., 0.])
  draw!(img_rhs, labels_rhs[5], labels_rhs[6], Float32[0., 0., 1.])

  return (img_lhs, img_rhs)
end
