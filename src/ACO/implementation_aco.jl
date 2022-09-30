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

# ╔═╡ d2c2971a-7160-4035-ad43-c6e7412ec249
md"""
## Imports
"""

# ╔═╡ d3349619-017d-4d16-aeae-6150f3533a39
md"""
## Structs
"""

# ╔═╡ c6cfd2c4-fa36-49b8-b054-5d198610c31d
struct ACOSettings
	α:: Real
	β:: Real
	number_of_ants:: Integer
	ρ:: Real
	ϵ:: Real
	max_number_of_iterations:: Integer
	starting_pheromone_ammount:: Real
	eval_f:: Function
	compute_solution:: Function
	ACOSettings(α, β, n_a, ρ, ϵ, max_i, start_ph) = new(α, β, n_a, ρ, ϵ, max_i, start_ph, (_, _) -> 1.0, (_, _) -> 1.0)
	ACOSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, e_f, c_s) = new(α, β, n_a, ρ, ϵ, max_i, start_ph, e_f, c_s)
end

# ╔═╡ 8c158b69-41cb-422f-a7df-10f390792c8f
mutable struct ACOInner
	graph
	n
	η
	τ
end

# ╔═╡ 8dd697a4-a690-4a99-95b5-8410756d4ba4
md"""
## Helper functions
"""

# ╔═╡ f60bc671-65a7-4335-8388-22535751d23b
sample(weights) = findfirst(cumsum(weights) .> rand())

# ╔═╡ 78a41b8e-18a8-45f3-85a8-693fe2a0b1c2
spread(inner::ACOInner) = inner.graph, inner.n, inner.η, inner.τ

# ╔═╡ 42bee75a-7831-4214-9eed-251f6c699832
md"""
## Solution generation functions
"""

# ╔═╡ adaaeb50-f117-45ff-934b-890be5e972fe
# Get chosen point
function get_chosen_point(pM, i, r)
	if maximum(pM[i, :]) == 0
		return i
	end

	findfirst(pM[i, :] .> r)
end

# ╔═╡ 467549ca-519a-4a84-98e7-9e78d93342a2
# Constructs a new solution
function generate_s(n::Integer, pM::Matrix{Float64})
	r = rand(n)
	
	[get_chosen_point(pM, i, r[i]) for i in 1:n]
end

# ╔═╡ ca49cc0e-b106-4dff-ad64-0ae5a568920c
# Constructs a new solution
function generate_s_avoid_duplicate(n, pM)
	s = zeros(Int32, n)

	for i in 1:n
		j = 0
		res = get_chosen_point(pM, i, rand())
		while s[res] == i && j < 100
			res = get_chosen_point(pM, i, rand())
			j += 1
		end
		s[i] = res
	end

	s
end

# ╔═╡ aa8314fd-ab17-4e23-9e83-dc79a6f69209
function choose_iteration_best(inner::ACOInner, settings::ACOSettings, iterations)
	filtered_iter = filter(x -> x != nothing, iterations)
	solutions = Folds.map(x -> settings.compute_solution(inner.n, x), filtered_iter)
	points = Folds.map(x -> settings.eval_f(inner.graph, x), solutions)
	index = argmax(points)
	(iterations[index], points[index])
end

# ╔═╡ c94724e8-f634-4506-a605-8d2dd68f8b0b
md"""
## Combination functions
"""

# ╔═╡ bbf1563c-6b72-40b6-accc-197a36ebe5c7
function get_combined_c(c, (c_dest, c_source))
	r = copy(c)

	r[r .== c_source] .= c_dest

	r
end

# ╔═╡ 4beb6af6-1dde-46f7-8efb-85d16e305243
function a_lt_b_not_empty(x, c)
	(a, b) = x

	if a >= b
		return false
	end

	count(c .== a) != 0 && count(c .== b) != 0
end

# ╔═╡ e004a0f1-99b1-465f-89b3-6e1cc324f560
function get_combined_result(g, eval_f, c, x)
	comb = get_combined_c(c, x)
	val = eval_f(g, comb)
	(val, x, comb)
end

# ╔═╡ 18d6233b-b7ea-4707-9cfb-18e1a3b19b46
function get_best_combination(graph, eval_f, c)
	n = maximum(c)

	joins_mat = [(i, j) for i in 1:n for j in 1:n]
	joins = filter(x -> a_lt_b_not_empty(x, c), joins_mat[:])

	joins_modularities = Folds.map(x -> get_combined_result(graph, eval_f, c, x), joins)

	if length(joins_modularities) == 0
		return c
	end

	best = argmax(joins_modularities)
	_, _, c_res = joins_modularities[best]

	c_res
end

# ╔═╡ 165073f5-cc4e-4f6b-bab7-088a569c385d
function number_of_communitites(c)
	com_num = map(x -> count(c .== x), 1:maximum(c))

	length(filter(x -> x > 0, com_num))
end

# ╔═╡ 4ee1d480-65b1-405c-8a21-7e88eed3fa92
function reduce_number_of_communities(graph, eval_f, c, n)
	if n == 0
		return c
	end

	iter = 0
	p = deepcopy(c)
	while number_of_communitites(p) > n && iter < 100
		p = get_best_combination(graph, eval_f, p)
		iter += 1
	end

	p
end

# ╔═╡ 43b8516e-ee3f-4463-8f4c-e0f7947c6df8
md"""
## Main functions
"""

# ╔═╡ 8167196c-4a45-45f0-b55b-26b69f27904b
function ACO(graph, vars::ACOSettings, η, τ; k = 0)
	#Set parameters and initialize pheromone traits.
	n, _ = size(η)
	inner = ACOInner(graph, n, η, τ)
	
	
	sgb = [i for i in 1:n]
	sgb_val = -1000
	τ_max = vars.starting_pheromone_ammount
	τ_min = 0

	# Precomputing this
	η_d_sq = inner.η .^ vars.β
	
	# While termination condition not met
	 for i in 1:vars.max_number_of_iterations
		# Construct new solution s according to Eq. 2

		# Precomputing the probabilities results in a 2s time improvement.
		probM = inner.τ .^ vars.α .* η_d_sq
		#probM = [inner.τ[i, j] ^ vars.α * inner.η[i, j] ^ vars.β for i in 1:n, j in 1:n]
		probM ./= sum(probM, dims=2)
		probM = cumsum(probM, dims=2)
		

		if i % 3 < 2
			S = Folds.map(x -> generate_s(n, probM), 1:vars.number_of_ants)
		else
			S = Folds.map(x -> generate_s_avoid_duplicate(n, probM), 1:vars.number_of_ants)
		end

		# Update iteration best
		(sib, sib_val) = choose_iteration_best(inner, vars, S)
		if sib_val > sgb_val
			sgb_val = copy(sib_val)
			sgb = copy(sib)
			
			# Compute pheromone trail limits
			τ_max = sgb_val / (1 - vars.ρ)
			τ_min = vars.ϵ * τ_max
		end
		# Update pheromone trails
		# TODO: test with matrix sum
		τ .*= (1 - vars.ρ)
		for (a, b) in enumerate(sib)
			# if sib[b] != a || a < b
				τ[a, b] += sib_val
				# τ[b, a] += sib_val
			# end
		end
		τ = min.(τ, τ_max)
		τ = max.(τ, τ_min)

	end
	
	reduce_number_of_communities(graph, vars.eval_f, vars.compute_solution(inner.n, sgb), k), τ
end

# ╔═╡ 8c6beaf2-f0e0-4d48-b593-06bcdba36455
function ACO(graph, vars::ACOSettings, η; k = 0)
	n, _ = size(η)
	τ = ones(n, n) .* vars.starting_pheromone_ammount
	r, _ = ACO(graph, vars, η, τ; k=k)

	r
end

# ╔═╡ 0f67bf52-8ff3-4ddc-a2c8-e02f48fb5f6c
function ACO_get_pheromone(graph, vars::ACOSettings, η; k=0)
	n, _ = size(η)
	τ = ones(n, n) .* vars.starting_pheromone_ammount
	ACO(graph, vars, η, τ; k=k)
end

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

julia_version = "1.7.2"
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
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "04d13bfa8ef11720c24e4d840c0033d145537df7"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.17"

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
# ╟─d2c2971a-7160-4035-ad43-c6e7412ec249
# ╠═57912b3a-83ae-11ec-0fbf-9da8ce954fe1
# ╟─d3349619-017d-4d16-aeae-6150f3533a39
# ╠═c6cfd2c4-fa36-49b8-b054-5d198610c31d
# ╠═8c158b69-41cb-422f-a7df-10f390792c8f
# ╟─8dd697a4-a690-4a99-95b5-8410756d4ba4
# ╠═f60bc671-65a7-4335-8388-22535751d23b
# ╠═78a41b8e-18a8-45f3-85a8-693fe2a0b1c2
# ╟─42bee75a-7831-4214-9eed-251f6c699832
# ╠═adaaeb50-f117-45ff-934b-890be5e972fe
# ╠═467549ca-519a-4a84-98e7-9e78d93342a2
# ╠═ca49cc0e-b106-4dff-ad64-0ae5a568920c
# ╠═aa8314fd-ab17-4e23-9e83-dc79a6f69209
# ╟─c94724e8-f634-4506-a605-8d2dd68f8b0b
# ╠═bbf1563c-6b72-40b6-accc-197a36ebe5c7
# ╠═4beb6af6-1dde-46f7-8efb-85d16e305243
# ╠═e004a0f1-99b1-465f-89b3-6e1cc324f560
# ╠═18d6233b-b7ea-4707-9cfb-18e1a3b19b46
# ╠═165073f5-cc4e-4f6b-bab7-088a569c385d
# ╠═4ee1d480-65b1-405c-8a21-7e88eed3fa92
# ╟─43b8516e-ee3f-4463-8f4c-e0f7947c6df8
# ╠═8167196c-4a45-45f0-b55b-26b69f27904b
# ╠═8c6beaf2-f0e0-4d48-b593-06bcdba36455
# ╠═0f67bf52-8ff3-4ddc-a2c8-e02f48fb5f6c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
