import FluxTraining
FluxTraining.stateaccess(::Type{<:LossParam}) = (lossfn = Write(), )
FluxTraining.sethyperparameter!(learner, t::Type{<:LossParam}, val) = begin
  let sym = string(t) |> lowercase |> Symbol
    setfield!(learner.lossfn, sym, val)
  end
  return learner
end

parse_defaults(val::Number) = val
parse_defaults(vec::Vector) = parse_defaults.(vec)
parse_defaults(str::String) = str
parse_defaults(f::Function) = String(nameof(f))
function parse_defaults(cfg_struct) :: Dict
  fields = fieldnames(typeof(cfg_struct))
  Dict( String(f) => parse_defaults(getfield(cfg_struct, f))
        for f in fields )
end
