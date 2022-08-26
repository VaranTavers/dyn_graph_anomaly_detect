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
	import .implementation_jl: ACO, ACOSettings, ACO_get_pheromone
end

# ╔═╡ 8dd697a4-a690-4a99-95b5-8410756d4ba4
md"""
## Helper functions
"""

# ╔═╡ 39bdd460-58de-4bcc-8237-12586b74a11b
function logistic(x)
	1 / (1 + exp(-x))
end

# ╔═╡ fefdd2c0-02a7-4db3-b868-f40c98373e1f
# Kronecker delta function
function δ(i, j)
	if i == j
		return 1
	end
	0
end

# ╔═╡ f644dc75-7762-4ae5-98f7-b5d9e5a05e39
md"""
## Benchmark functinos
"""

# ╔═╡ c88acc56-32a0-4209-aa92-eec60e1bbbe7
md"""
$A_{ij} - \frac{k_i * k_j}{2m}$
"""

# ╔═╡ bf2fe679-e748-4117-9425-c4285f0d729e
function calculate_modularity_inner(graph, i, j)
	begin
	m = ne(graph)
	A_ij = has_edge(graph, i, j) ? 1 : 0
	k_i = length(all_neighbors(graph, i))
	k_j = length(all_neighbors(graph, j))

	A_ij - (k_i * k_j) / 2m
	end
end

# ╔═╡ 74f178c1-4ecd-4ecf-8a08-ce3a4dbdb766
function calculate_modularity(graph, c)
	begin
	m = ne(graph)
	n = nv(graph)
	numbers = collect(1:n)

	rows = Folds.map(i -> [calculate_modularity_inner(graph, i, j) for j in numbers[c .== c[i]]], 1:n)
	s = sum(sum.(rows))

	s / 2m
	end
end

# ╔═╡ 0d55705f-b656-4479-a7e9-5cbfef9063b3
function community_entropy_inner(c, i)
	n = length(c)
	nx_i = count(x -> x == i, c)

	nx_i * log(nx_i / n) / n
end

# ╔═╡ af25b3e3-fee0-4cf3-bbe9-8b5b7202926e
function community_entropy(c)
	- sum([community_entropy_inner(c, i) for i in 1:maximum(c)])
end

# ╔═╡ 31115dfc-2898-431e-962a-588e854a05d8
function mutual_information_inner(c1, c2, i, j)
	n = length(c1)
	z = zip(c1, c2)
	nxy_ij = count(x -> x == (i, j), z)
	nx_i = count(x -> x == i, c1)
	ny_j = count(x -> x == j, c2)

	if nxy_ij == 0
		return 0
	end
	
	nxy_ij * log((nxy_ij / n) / ((nx_i / n) * (ny_j / n))) / n
end

# ╔═╡ c833cd39-58a3-4a8c-8281-d8ec862a0314
function mutual_information(c1, c2)
	sum([sum([mutual_information_inner(c1, c2, i, j) for j in 1:maximum(c2)]) for i in 1:maximum(c1)])
end

# ╔═╡ b6b71f29-5929-4b1d-abb4-e182deb5c8b3
function normalized_mutual_information(c1, c2)
	2 * mutual_information(c1, c2) / (community_entropy(c1) + community_entropy(c2))
end

# ╔═╡ 8b2e3cd3-fb5d-4a81-8cf4-27b956088bab
md"""
## Pearson Corelation 
"""

# ╔═╡ e66b3fe1-7979-4811-a349-4b027e112310
md"""
$C(i,j) = \frac{\sum_{v_t \in V}{(A_{il} - \mu_i)(A_{jl} - \mu_j)}}{n\sigma_i\sigma_j}$
"""

# ╔═╡ 0bbaaf25-4633-4a82-859e-db81068d680a
function pearson_corelation(graph::SimpleWeightedGraph{Int64, Float64}, i, j)
	n = nv(graph)
	
	μ_i = sum(graph.weights[i, :]) / n
	μ_j = sum(graph.weights[j, :]) / n
	σ_i = sqrt(sum((graph.weights[i, :] .- μ_i) .^ 2) / n)
	σ_j = sqrt(sum((graph.weights[j, :] .- μ_j) .^ 2) / n)

	if σ_i * σ_j == 0
		return -1
	end

	numerator = sum((graph.weights[i, :] .- μ_i) .* (graph.weights[j, :] .- μ_j))

	numerator / (n * σ_i * σ_j)
end

# ╔═╡ 8df82f08-118b-406a-a0a4-c5699590f9df
# Built for bidirectional edges
# Transforms the edge representation from generate_s to a community vector.
function compute_solution(n, edges)
	begin
	tmp_g = SimpleWeightedGraph(n)
	for (a, b) in enumerate(edges)
		add_edge!(tmp_g, a, b)
	end
	
	s = zeros(Int32, n)
	start = 1
	clust = 1
	while start <= n
		# Skip the points, which already have a community
		while start <= n && s[start] != 0
			start += 1
		end
		if start > n
			break
		end
		# If we have found a point that has no community we check it has any neighbors. If it doesn't we don't assign it to a community. (it has no neighbors if it has an edge to itself)
		if tmp_g.weights[start, start] == 0
			s[start] = clust
			for (j, v) in enumerate(dfs_parents(tmp_g, start))
				if v > 0
					s[j] = clust
				end
			end
			clust += 1
		else
			start += 1
		end
	end
	
	s
	end
end

# ╔═╡ 4c4ae7b7-04c3-41a3-ba91-3db1f6522ea2
function copy_replace_funcs(vars_base::ACOSettings, eval_f, c_s)
	ACOSettings(
		vars_base.α,
		vars_base.β,
		vars_base.number_of_ants,
		vars_base.ρ,
		vars_base.ϵ,
		vars_base.max_number_of_iterations,
		vars_base.starting_pheromone_ammount,
		eval_f,
		c_s
	)
end

# ╔═╡ 481e0c2b-00f6-408c-b829-1872415892e7
function same_neighbors(g, i, j)
	a = Set(all_neighbors(g, i))
	b = Set(all_neighbors(g, j))
	
	length(intersect(a, b)) / nv(g)
end

# ╔═╡ 64a08561-b1c3-4d42-b58f-85bb8486460b
# We can calculate the number of common neighbors faster if we multiply the incidency matrix with it's tranposed matrix.
function get_η_common_neighbors(g2)
	M = ceil.(abs.(g2.weights))
	M2 = (M * M' .+ 1) / nv(g2)
	[i != j ? (g2.weights[i, j] != 0 ? M2[i, j] : 0.0) : 0.00001 for i in 1:nv(g2), j in 1:nv(g2)]
end

# ╔═╡ a42d4687-b1d2-4a9d-b882-7d648422a72c
function CommunityACO(graph, vars_base::ACOSettings, τ; k=0)
	n = nv(graph)
	η = get_η_common_neighbors(graph)

	vars = copy_replace_funcs(vars_base, calculate_modularity, compute_solution)

	ACO(graph, vars, η, τ; k)
end

# ╔═╡ 29da4832-6d05-4dfa-8be9-0dd01893ede1
function CommunityACO(graph, vars_base::ACOSettings; k=0)
	n = nv(graph)
	η = get_η_common_neighbors(graph)

	vars = copy_replace_funcs(vars_base, calculate_modularity, compute_solution)

	ACO(graph, vars, η; k)
end

# ╔═╡ 72a81225-6ecd-4ae6-b668-1ad4af0d6b7c
function CommunityACO_get_pheromone(graph, vars_base::ACOSettings; k=0)
	n = nv(graph)
	η = get_η_common_neighbors(graph)

	vars = copy_replace_funcs(vars_base, calculate_modularity, compute_solution)
	
	ACO_get_pheromone(graph, vars, η; k=k)
end

# ╔═╡ e8f705bb-be85-48aa-a25e-0f9e2921f6a3
begin
	g = loadgraph("../../graphs/changhonghao2013 (copy).lgz", SWGFormat())
	
	vars = ACOSettings(
			1, # α
			2, # β
			30, # number_of_ants
			0.8, # ρ
			0.005, # ϵ
			100, # max_number_of_iterations
			3 # starting_pheromone_ammount
		)
	c = CommunityACO(g, vars)
	c
	calculate_modularity(g, c)
	
end

# ╔═╡ 18788290-bc3a-4ef9-ae9b-9034957228f5
begin
	g2 = loadgraph("../../graphs/LFR/network6.lgz", SWGFormat())
	
	vars2 = ACOSettings(
			1, # α
			2, # β
			30, # number_of_ants
			0.8, # ρ
			0.005, # ϵ
			100, # max_number_of_iterations
			3 # starting_pheromone_ammount
		)
	c2, phe = CommunityACO_get_pheromone(g2, vars2)
	c2
	calculate_modularity(g2, c2)
	
end

# ╔═╡ 750d6c6d-9760-4914-b22a-a841b994ca4a
phe[7, :]

# ╔═╡ 5a70e933-f62e-4008-a683-a36e58b7da6b
begin
	g3 = loadgraph("../../graphs/dynamic_graphs/syntetic/school_test/school_test52.lgz", SWGFormat())
	
	@time c3 = CommunityACO(g3, vars2)
	calculate_modularity(g3, c3)
	
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
# ╟─8dd697a4-a690-4a99-95b5-8410756d4ba4
# ╠═39bdd460-58de-4bcc-8237-12586b74a11b
# ╠═fefdd2c0-02a7-4db3-b868-f40c98373e1f
# ╟─f644dc75-7762-4ae5-98f7-b5d9e5a05e39
# ╟─c88acc56-32a0-4209-aa92-eec60e1bbbe7
# ╠═bf2fe679-e748-4117-9425-c4285f0d729e
# ╠═74f178c1-4ecd-4ecf-8a08-ce3a4dbdb766
# ╠═0d55705f-b656-4479-a7e9-5cbfef9063b3
# ╠═af25b3e3-fee0-4cf3-bbe9-8b5b7202926e
# ╠═31115dfc-2898-431e-962a-588e854a05d8
# ╠═c833cd39-58a3-4a8c-8281-d8ec862a0314
# ╠═b6b71f29-5929-4b1d-abb4-e182deb5c8b3
# ╟─8b2e3cd3-fb5d-4a81-8cf4-27b956088bab
# ╟─e66b3fe1-7979-4811-a349-4b027e112310
# ╠═0bbaaf25-4633-4a82-859e-db81068d680a
# ╠═8df82f08-118b-406a-a0a4-c5699590f9df
# ╠═4c4ae7b7-04c3-41a3-ba91-3db1f6522ea2
# ╠═481e0c2b-00f6-408c-b829-1872415892e7
# ╠═64a08561-b1c3-4d42-b58f-85bb8486460b
# ╠═a42d4687-b1d2-4a9d-b882-7d648422a72c
# ╠═29da4832-6d05-4dfa-8be9-0dd01893ede1
# ╠═72a81225-6ecd-4ae6-b668-1ad4af0d6b7c
# ╠═e8f705bb-be85-48aa-a25e-0f9e2921f6a3
# ╠═18788290-bc3a-4ef9-ae9b-9034957228f5
# ╠═750d6c6d-9760-4914-b22a-a841b994ca4a
# ╠═5a70e933-f62e-4008-a683-a36e58b7da6b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
