"Creates experiment path based on current git hash, a git diff hash and an increasing version number
 A git diff file is also stored within the created directory."
function make_experiment_path()
  current_branch = readchomp(`git branch --show-current`)  # empty when in detached head state
  commit_hash = readchomp(`git rev-parse --short HEAD`)
  diff_hash = readchomp(pipeline(`git diff`, `sha1sum`, `head -c 8`))

  unversioned_experiment_path = joinpath("experiments", "$(current_branch)_$(commit_hash)_$(diff_hash)")
  if !ispath(unversioned_experiment_path)
    mkdir(unversioned_experiment_path)
    write(joinpath(unversioned_experiment_path, "diff.txt"), read(`git diff`))
  end
  new_version = 
    try
      maximum([parse(Int, match(r"version_([0-9]+)", d)[1]) for d in readdir(unversioned_experiment_path) if occursin("version_", d)]) + 1
    catch ArgumentError  # if there is no directory
      1
    end
  versioned_experiment_path = joinpath(unversioned_experiment_path, "version_$(new_version)")
  mkdir(versioned_experiment_path)
end
