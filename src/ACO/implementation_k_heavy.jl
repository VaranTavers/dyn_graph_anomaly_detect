### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 57912b3a-83ae-11ec-0fbf-9da8ce954fe1
begin
	using Graphs
	using Distributed
	using CSV
	using DataFrames
	using SimpleWeightedGraphs
	using Folds
end

# ╔═╡ d8470914-6064-43a2-9dec-e32302b0fb80
md"""
## Imports
"""

# ╔═╡ db5b170a-f3f8-4667-a65c-1f3285d0275c
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

# ╔═╡ 8fa16cd8-1735-4200-9ad0-4cdbe3e71c91
begin
	implementation_jl = ingredients("./implementation_aco.jl")
	import .implementation_jl: sample, ACOSettings, ACOInner, spread
end

# ╔═╡ c9d3fbd8-773f-43ab-bd54-d36c2d16a525
struct ACOKSettings 
	acos # :: ACOSettings
	k:: Real
	# There are situations when the ACO algorithm is unable to create the k subgraph
	# There is two options there:
	# 	false - skip the solution (faster, but might give worse answers, this is recommended if you have points with no neighbours)
	# 	true  - regenerate solution until a possible one is created (slower, but might give better answers)
	force_every_solution:: Bool
	# If we force the length to be exactly k we may not get the correct answer since if we enter into a point from which only negative edges lead to anywhere the algorithm must choose a negative lenght even if there is a way to get to that point using an other route going back to where we come from. Now we can't let these answers become too big, since that would lead to infinite loops, so you can set an upper bound here:
	# If this equals k we will use the older method with the afforementioned error.
	solution_max_length:: Integer
	ACOKSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, k, f, s) = new(ACOSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, (_, _) -> 1.0, (_, _) -> 1.0), k, f, s)
	ACOKSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, e_f, c_s, k, f, s) = new(ACOSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, e_f, c_s), k, f, s)
	ACOKSettings(acos, k, f, s) = new(acos, k, f, s)
end

# ╔═╡ 8dd697a4-a690-4a99-95b5-8410756d4ba4
md"""
## Solution generator functions
"""

# ╔═╡ 0fe2aa25-71b5-41e1-bdd7-3d74cd2c7afe
# Calculates the probabilities of choosing edges to add to the solution.
function calculate_probabilities_old(inner::ACOInner, i, vars::ACOSettings, c)
	graph, n, η, τ = spread(inner)

	# graph.weights[i,j] * 
	p = [ findfirst(x -> x == j, c) == nothing ? (τ[i, j]^vars.α * η[i, j]^vars.β) : 0 for j in 1:n]
	if maximum(p) == 0
		p[i] = 1
	end
	s_p = sum(p)

	p ./= s_p

	p
end

# ╔═╡ e2523de6-262d-47ee-8168-6cd7b2aab6b7
# Calculates the probabilities of choosing edges to add to the solution.
function calculate_probabilities(inner::ACOInner, i, vars::ACOSettings)
	graph, n, η, τ = spread(inner)

	# graph.weights[i,j] * 
	p = [ (τ[i, j]^vars.α * η[i, j]^vars.β) for j in 1:n]

	if maximum(p) == 0
		p[i] = 1
	end
	s_p = sum(p)

	p ./= s_p

	p
end

# ╔═╡ 0ec92459-f5d7-4a0c-9ace-9324997dff17
function generate_s(inner::ACOInner, vars::ACOKSettings, i)
	if vars.k == vars.solution_max_length
		return generate_s_old(inner, vars)
	end
	
	points = zeros(Int64, vars.solution_max_length + 1)
	points[1] = i
	j = 2
	while length(unique(points)) - 1 < vars.k && j <= vars.solution_max_length
		
		points[j] = sample(calculate_probabilities(inner, points[j - 1], vars.acos))
		j += 1
	end

	if j == vars.solution_max_length + 1 && length(unique(points)) - 1 < vars.k
		if vars.force_every_solution
			return generate_s(inner, vars, i)
		else
			return
		end
	end

	points[1:(j-1)]
end

# ╔═╡ 7e986200-7274-4be5-9b18-3a0b84b570af
# Constructs a new solution, the old way
function generate_s_old(inner::ACOInner, vars::ACOKSettings)
	i = rand(1:inner.n)
	points = zeros(Int64, vars.k)
	points[1] = i
	for i in 2:vars.k
		points[i] = sample(calculate_probabilities_old(inner, points[i - 1], vars.acos, points))
		if points[i] == points[i - 1]
			if vars.force_every_solution
				return generate_s(inner, vars)
			else
				return
			end
		end
	end

	points
end

# ╔═╡ 6da30365-bb02-47fc-a812-b1f3571a909e
function choose_iteration_best(inner::ACOInner, settings::ACOSettings, iterations)
	iterations = filter(x -> x != nothing, iterations)
	points = Folds.map(x -> settings.eval_f(inner.graph, settings.compute_solution(inner.graph, x)), iterations)
	index = argmax(points)
	(iterations[index], points[index])
end

# ╔═╡ f644dc75-7762-4ae5-98f7-b5d9e5a05e39
md"""
## Benchmark functinos
"""

# ╔═╡ d8bfc438-2d64-4d2f-a66f-14fad5fcaf76
begin
	fst((a, _)) = a
	snd((_, b)) = b
end

# ╔═╡ 200c98de-e505-405e-ab2a-53eefd3fb2fa
function calculate_η_ij(graph, i, j, m)
	if graph.weights[i, j] == 0
		return 0;
	end

	graph.weights[i, j] - m + 1 + sum(graph.weights[:, j])
end

# ╔═╡ a854518f-2b68-402c-a754-c20000504f0a
function calculate_η(graph)
	n = nv(graph)

	m = minimum(graph.weights)
	if m >= 0
		m = 0
	end
	η = [ calculate_η_ij(graph, i, j, m) for i in 1:n, j in 1:n]

	η
end

# ╔═╡ a828f66b-b4cf-4ce9-953f-fd6499b5cda2
function get_weight(g, (x,y))
	g.weights[x, y]
end

# ╔═╡ b48ba4f0-f409-43c2-bde2-cb90acd5085d
function calculate_heaviness(graph, c)
	sum(map(x -> get_weight(graph, x) , c))
end

# ╔═╡ 705178d4-af4b-4104-a660-1de2ba77e81a
function compute_solution(graph, s)
	edges_orig = [(0,0) for i in 1:(length(s)-1)]
	for i in 1:(length(s) - 1)
		if s[i] < s[i+1]
			edges_orig[i] = (s[i], s[i+1])
		else
			edges_orig[i] = (s[i+1], s[i])
		end
	end
	edges_else = []
	for i in 1:(length(s) - 2)
		for j in (i+2):length(s)
			if graph.weights[s[i], s[j]] > 0
				if s[i] < s[j]
					push!(edges_else, (s[i], s[j]))
				else
					push!(edges_else, (s[j], s[i]))
				end
			end
		end
	end
	append!(edges_orig, edges_else)
	
	unique(edges_orig)
end

# ╔═╡ c9c59dc9-ecf6-4442-a867-a8c63120b382
function ACOK(graph, vars::ACOKSettings, η, τ)
	#Set parameters and initialize pheromone traits.
	n, _ = size(η)
	inner = ACOInner(graph, n, η, τ)
	
	@assert nv(graph) >= vars.k	
	@assert vars.k <= vars.solution_max_length
	sgb = [i for i in 1:n]
	sgb_val = -1000
	τ_max = vars.acos.starting_pheromone_ammount
	τ_min = 0
	
	# While termination condition not met
	for i in 1:vars.acos.max_number_of_iterations
		# Construct new solution s according to Eq. 2

		if i < vars.acos.max_number_of_iterations - 3
			S = Folds.map(x -> generate_s(inner, vars, rand(1:inner.n)), zeros(vars.acos.number_of_ants))
		else
			S = Folds.map(x -> generate_s(inner, vars, x), 1:inner.n)
		end

		if length(filter(x -> x != nothing, S)) > 0
			# Update iteration best
			(sib, sib_val) = choose_iteration_best(inner, vars.acos, S)
			if sib_val > sgb_val
				sgb_val = sib_val
				sgb = sib
				
				# Compute pheromone trail limits
				τ_max = sgb_val / (1 - vars.acos.ρ)
				τ_min = vars.acos.ϵ * τ_max
			end
			# Update pheromone trails
			# TODO: test with matrix sum
			τ .*= (1 - vars.acos.ρ)
			for (a, b) in zip(sib, sib[2:end])
				τ[a, b] += sib_val
				τ[b, a] += sib_val
	
			end
		end
		τ = min.(τ, τ_max)
		τ = max.(τ, τ_min)

	end
	
	vars.acos.compute_solution(inner.graph, sgb), τ
end

# ╔═╡ 7abb51b8-f103-4e18-b929-df1d7e22f70b
function ACOK(graph, vars::ACOKSettings, η)
	n, _ = size(η)
	τ = ones(n, n) .* vars.acos.starting_pheromone_ammount
	r, _ = ACOK(graph, vars, η, τ)

	r
end


# ╔═╡ bf725344-ae9a-41e7-8378-90be50e04b2b
function ACOK_get_pheromone(graph, vars::ACOKSettings, η)
	n, _ = size(η)
	τ = ones(n, n) .* vars.acos.starting_pheromone_ammount
	ACOK(graph, vars, η, τ)
end

# ╔═╡ 4c4ae7b7-04c3-41a3-ba91-3db1f6522ea2
function copy_replace_funcs(vars_base::ACOKSettings, eval_f, c_s)
	ACOKSettings(
		ACOSettings(
			vars_base.acos.α,
			vars_base.acos.β,
			vars_base.acos.number_of_ants,
			vars_base.acos.ρ,
			vars_base.acos.ϵ,
			vars_base.acos.max_number_of_iterations,
			vars_base.acos.starting_pheromone_ammount,
			eval_f,
			c_s
		),
		vars_base.k,
		vars_base.force_every_solution,
		vars_base.solution_max_length,
	)
end

# ╔═╡ a42d4687-b1d2-4a9d-b882-7d648422a72c
function HeaviestACOK(graph, vars_base::ACOKSettings, τ)
	η = calculate_η(graph)

	vars = copy_replace_funcs(vars_base, calculate_heaviness, compute_solution)

	ACOK(graph, vars, η, τ)
end

# ╔═╡ 29da4832-6d05-4dfa-8be9-0dd01893ede1
function HeaviestACOK(graph, vars_base::ACOKSettings)
	η = calculate_η(graph)

	vars = copy_replace_funcs(vars_base, calculate_heaviness, compute_solution)

	ACOK(graph, vars, η)
end

# ╔═╡ 72a81225-6ecd-4ae6-b668-1ad4af0d6b7c
function HeaviestACOK_get_pheromone(graph, vars_base::ACOKSettings)
	η = calculate_η(graph)

	vars = copy_replace_funcs(vars_base, calculate_heaviness, compute_solution)
	
	ACOK_get_pheromone(graph, vars, η)
end

# ╔═╡ 7e37429c-521d-4d6a-bebd-475cb9573803
function solution_to_community(graph, solution)
	n = nv(graph)

	f = fst.(solution)
	s = snd.(solution)
	append!(f, s)

	[ findfirst(x->x==i, f) != nothing ? 1 : 0 for i in 1:n]
end

# ╔═╡ e8f705bb-be85-48aa-a25e-0f9e2921f6a3
begin
	g = loadgraph("../../graphs/heavy/changhonghao.lgz", SWGFormat())
	
	vars = ACOSettings(
			1, # α
			2, # β
			30, # number_of_ants
			0.8, # ρ
			0.005, # ϵ
			100, # max_number_of_iterations
			3 # starting_pheromone_ammount
		)
	c = HeaviestACOK(g, ACOKSettings(vars, 12, false, 20))
	calculate_heaviness(g, c)
	
end

# ╔═╡ 83ec8e2b-3f89-46f1-b70e-b53c51496364
solution_to_community(g, c)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
Folds = "41a02a25-b8f0-4f67-bc48-60067656b558"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622"

[compat]
CSV = "~0.10.2"
DataFrames = "~1.3.2"
Folds = "~0.2.7"
Graphs = "~1.5.1"
SimpleWeightedGraphs = "~1.2.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Future", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "2bba2aa45df94e95b1a9c2405d7cfc3d60281db8"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.9"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "f87e559f87a45bece9c9ed97458d3afe98b1ebb9"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.1.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "d648adb5e01b77358511fb95ea2e4d384109fac9"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.35"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "9519274b50500b8029973d241d32cfbf0b127d97"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ae02104e835f219b8930c7664b8012c93475c340"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.2"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "04d13bfa8ef11720c24e4d840c0033d145537df7"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.17"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.Folds]]
deps = ["Accessors", "BangBang", "Baselet", "DefineSingletons", "Distributed", "InitialValues", "MicroCollections", "Referenceables", "Requires", "Test", "ThreadedScans", "Transducers"]
git-tree-sha1 = "8559de3011264727473c96e1f794f9ddcac2bb1c"
uuid = "41a02a25-b8f0-4f67-bc48-60067656b558"
version = "0.2.7"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "d727758173afef0af878b29ac364a0eca299fc6b"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.5.1"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "13468f237353112a01b2d6b32f3d0f80219944aa"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "e681d3bfa49cd46c3c161505caddf20f0e62aaa9"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "6a2f7d70512d205ca8c7ee31bfa9f142fe74310c"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.12"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "0afd9e6c623e379f593da01f20590bacc26d1d14"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "a635a9333989a094bddc9f940c04c549cd66afcf"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadedScans]]
deps = ["ArgCheck"]
git-tree-sha1 = "ca1ba3000289eacba571aaa4efcefb642e7a1de6"
uuid = "24d252fe-5d94-4a69-83ea-56a14333d47a"
version = "0.1.0"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "1cda71cc967e3ef78aa2593319f6c7379376f752"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.72"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "c69f9da3ff2f4f02e811c3323c22e5dfcb584cfa"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─d8470914-6064-43a2-9dec-e32302b0fb80
# ╠═57912b3a-83ae-11ec-0fbf-9da8ce954fe1
# ╠═db5b170a-f3f8-4667-a65c-1f3285d0275c
# ╠═8fa16cd8-1735-4200-9ad0-4cdbe3e71c91
# ╠═c9d3fbd8-773f-43ab-bd54-d36c2d16a525
# ╟─8dd697a4-a690-4a99-95b5-8410756d4ba4
# ╠═0fe2aa25-71b5-41e1-bdd7-3d74cd2c7afe
# ╠═e2523de6-262d-47ee-8168-6cd7b2aab6b7
# ╠═7e986200-7274-4be5-9b18-3a0b84b570af
# ╠═0ec92459-f5d7-4a0c-9ace-9324997dff17
# ╠═6da30365-bb02-47fc-a812-b1f3571a909e
# ╟─f644dc75-7762-4ae5-98f7-b5d9e5a05e39
# ╠═d8bfc438-2d64-4d2f-a66f-14fad5fcaf76
# ╠═200c98de-e505-405e-ab2a-53eefd3fb2fa
# ╠═a854518f-2b68-402c-a754-c20000504f0a
# ╠═a828f66b-b4cf-4ce9-953f-fd6499b5cda2
# ╠═b48ba4f0-f409-43c2-bde2-cb90acd5085d
# ╠═705178d4-af4b-4104-a660-1de2ba77e81a
# ╠═c9c59dc9-ecf6-4442-a867-a8c63120b382
# ╠═7abb51b8-f103-4e18-b929-df1d7e22f70b
# ╠═bf725344-ae9a-41e7-8378-90be50e04b2b
# ╠═4c4ae7b7-04c3-41a3-ba91-3db1f6522ea2
# ╠═a42d4687-b1d2-4a9d-b882-7d648422a72c
# ╠═29da4832-6d05-4dfa-8be9-0dd01893ede1
# ╠═72a81225-6ecd-4ae6-b668-1ad4af0d6b7c
# ╠═7e37429c-521d-4d6a-bebd-475cb9573803
# ╠═e8f705bb-be85-48aa-a25e-0f9e2921f6a3
# ╠═83ec8e2b-3f89-46f1-b70e-b53c51496364
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
