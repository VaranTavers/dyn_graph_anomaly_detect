### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 8049a0d0-7c45-11ed-1a8e-1135d3f81388
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ 0b0d07a7-8ff5-4aaa-905a-485d5677889a
x = 3

# ╔═╡ b505bb06-cf50-4ff2-9b72-a89fbb0052c9
begin
	if x == 2
		implementation_jl = ingredients("./ACO/implementation_acok_mindp2.jl")
		import .implementation_jl: ACOK, ACOKSettings, ACOK_get_pheromone, copy_replace_funcs, ACOKInner
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╠═8049a0d0-7c45-11ed-1a8e-1135d3f81388
# ╠═0b0d07a7-8ff5-4aaa-905a-485d5677889a
# ╠═b505bb06-cf50-4ff2-9b72-a89fbb0052c9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
