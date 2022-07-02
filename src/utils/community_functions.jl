### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 7e546c44-4e50-412f-9144-bb2056d749ff
begin
	using Folds
	using Statistics
end

# ╔═╡ 91d589d0-a609-11ec-2dbe-bb52470686eb
function calculate_community_similarity(c1, c2, i, j)
	n = length(c1)
	1 - count((c1 .== i) .⊻ (c2 .== j)) / n
end

# ╔═╡ 5abe5bbb-5ed5-4ff8-8ac6-4b253e4c82d1
function calculate_community_similarity2(c1, c2, i, j)
	n = length(c1)
	n1 = count((c1 .== i))

	# If a point is present in both communities 4 points
	# If a point is absent from both communities 2 points
	# If a point is present only in the second community 1 point
	(count((c1 .== i) .&& (c2 .== j)) * 4 -
	count((c1 .!= i) .&& (c2 .== j))) / (4 * n1)
end

# ╔═╡ a93406c3-db78-403d-bc7f-3d6de25ce608
function calculate_community_similarity3(c1, c2, i, j)
	n = length(c1)
	1 - count((c1 .== i) .⊻ (c2 .== j)) / count((c1 .== i) .|| (c2 .== j))
end

# ╔═╡ 79511d3a-1e2c-4aea-9889-1c5c8844746e
function calculate_community_similarity4(c1, c2, i, j)
	n = length(c1)
	n1 = count((c1 .== i))

	sim = (count((c1 .== i) .&& (c2 .== j)) - count((c1 .== i) .&& (c2 .!= j))) * 2
	- (count((c1 .!= i) .&& (c2 .!= j)) - count((c1 .!= i) .&& (c2 .== j)))

	1 / (1 + exp(-sim))
end

# ╔═╡ 1eb5bc1b-52d5-4482-a2cf-e83b044b15f5
function calc_sim(c_list, (com_i, c_list_i), (com_j, c_list_j), s_f)
	s_f(c_list[c_list_i], c_list[c_list_j], com_i, com_j)
end

# ╔═╡ 926c5c37-b87b-480e-b042-f89b7c4b456d
function relabel_communities(c_list, p; changing=false, similarity_f=calculate_community_similarity2)
	ret = deepcopy(c_list)
	communities = [1 for i in 1:maximum(c_list[1])]
	
	for i in 2:length(ret)
		c = ret[i]
		old_l = length(communities)
		c[c .> 0] .+= old_l
		n_c = maximum(c)
		translation = zeros(n_c)
		for c_i in (old_l + 1):n_c
			scores = Folds.map(x -> calc_sim(ret, x, (c_i, i), similarity_f), enumerate(communities))
			max_score_index = argmax(scores)
			if scores[max_score_index] < p
				push!(communities, i)
				c[c .== c_i] .= length(communities)
			else
				c[c .== c_i] .= max_score_index
				if changing
					communities[max_score_index] = i
				end
			end
		end
	end

	ret
end

# ╔═╡ 10041156-7aba-4fb9-a172-e7fe215fba8a
function calculate_community_activation(size_list)
	for (i, n) in enumerate(size_list)
		# If a community is present at this point, it barely appears before this point, and it is barely missing after this point, then we consider this point the activation of the community.
		if n != 0 && 
		count(size_list[i:end] .> 0) / (length(size_list[i:end])) > 0.8 &&	
		count(size_list[1:i] .== 0) / (i + 1) > 0.7
			return i
		end
	end

	0
end

# ╔═╡ 783d617d-113e-4ec9-a144-6eb060738f3d
function calculate_community_creation(size_list)
	findfirst(x -> x > 0, size_list)
end

# ╔═╡ 5be87169-43f1-4ddf-9c5c-9af5d4504804
function calculate_community_deactivation(size_list)
	for (i, n) in enumerate(size_list)
		# If a community is not present at this point, it barely appears after this point, and it was barely missing before this point, then we consider this point the deactivation of the community.
		if n == 0 && 
		count(size_list[i:end] .== 0) / (length(size_list[i:end])) > 0.8 &&	
		count(size_list[1:i] .> 0) / (i + 1) > 0.7
			return i
		end
	end

	0
end

# ╔═╡ c8a96dcb-65ae-4562-a05d-a6d286d626cc
function calculate_community_disappearance(size_list)
	findlast(x -> x > 0, size_list)
end

# ╔═╡ d169f9fd-fd0b-4209-967e-fd834439cb28
function calculate_unusual_size_change(c_s, baseline)
	[
		(i - baseline) / baseline > 1 ? 1 :
		(baseline - i) / baseline > 0.5 && i != 0 ? -1 : 0
		for i in c_s
	]
end

# ╔═╡ bdc427d8-7b17-4935-9995-2dd7e1937658
function calculate_unusual_size_change(c_s)
	baseline = Statistics.mean(filter(x -> x > 0, c_s))
	[
		i == 0 ? 0 : (
			(i - baseline) / baseline > 1 ? 1 :
			(baseline - i) / baseline > 0.5 && i != 0 ? -1 : 0
		)
		for i in c_s
	]
end

# ╔═╡ 8d11583f-eccf-45e0-8624-5f5158faf632
function calculate_unusual_appearance(c_s, birth, death)
	# If a community is not present in more than 10% of the point in time, it's every appearance is considered unusual, otherwise every appearance before "birth" or after "death" is considered unusual.
	if death == 0
		death = length(c_s) + 1
	end
	if count(c_s .!= 0) / length(c_s) < 0.1
		birth = length(c_s)
	end

	[ (i < birth && v > 0) || (i > death && v != 0) for (i, v) in enumerate(c_s)]
end

# ╔═╡ 629914f9-6411-417f-9eb8-b812a47551a1
function get_renamed_community(community, c_start, c_dest)
	res = deepcopy(community)
	res[res .== c_start] .= c_dest

	res
end

# ╔═╡ b92c7770-e5f0-4818-8b68-ee685e50123d
# Function to calculate if in 2 community lists ceartain communities merged / or were created as a result of the split of a larger community
# Split: c1 = prev_c; c2 = current_c, time = current_time - 1
# Merge: c2 = prev_c; c1 = current_c, time = current_time

function get_best_composition(c_1, c_2, time, size_lists, goal_c)
	communities_present = unique(c_2)
	filter0 = filter(x -> x != 0, communities_present)
	filter1 = filter(x -> size_lists[x][time] == 0, filter0)
	mapped = map(x -> (calculate_community_similarity2(c_1,
		c_2, 
		goal_c, 
		x),
	x), filter1)

	first((v, x)) = v
	second((v, x)) = x
	filter2 = filter(x -> first(x) > 0.15, mapped)

	communities_chosen = collect(filter2)
	values = map(first, communities_chosen)
	if sum(values) > 0.70 && length(values) > 1
		return collect(map(second, communities_chosen))
	end

	[]
end

# ╔═╡ 85ef9af4-4397-47ca-8ecf-5d1aa9d2193d
function calculate_splitting(communities_pred, size_lists, c_i)
	result = []
	size_list = size_lists[c_i]
	
	for (i, (a, b)) in enumerate(zip(size_list[2:end], size_list))
		if a != 0 || b == 0
			continue
		end
		res = get_best_composition(communities_pred[i], communities_pred[i + 1], i, size_lists, c_i)
		if length(res) > 0
			push!(result, (i + 1, res))
		end
	end

	result
end

# ╔═╡ cada7919-cfc6-46ae-a438-025dbef90b2a
function calculate_merging(communities_pred, size_lists, c_i)
	result = []
	size_list = size_lists[c_i]
	
	for (i, (a, b)) in enumerate(zip(size_list[2:end], size_list))
		if a == 0 || b != 0
			continue
		end
		res = get_best_composition(communities_pred[i + 1], communities_pred[i], i + 1, size_lists, c_i)
		if length(res) > 0
			push!(result, (i + 1, res))
		end
	end

	result
end

# ╔═╡ 04037256-85ac-4064-96ed-c9183ff34b1f
test_change_commnunities = [[1, 1, 1, 2, 2, 2], [1, 1, 1, 1, 2, 2], [2, 1, 1, 1, 1, 2], [2, 2, 1, 1, 1, 1]]

# ╔═╡ 7c10c2f1-158a-4158-8467-a8be9574460a
relabel_communities(test_change_commnunities, 0.3; changing=true, similarity_f=calculate_community_similarity3)

# ╔═╡ ee187301-524f-4452-be96-bd94e68aa4f9
relabel_communities(test_change_commnunities, 0.6; changing=false)

# ╔═╡ 1c3e0b4a-0447-4298-99c3-d5ef57e6b16a
test_split_communities = [[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[2, 3, 2, 3, 2, 3, 2, 3],[2, 3, 2, 3, 2, 3, 2, 3]]

# ╔═╡ 1724765f-5ba7-41b3-90af-70fc890e0053
test_split_size_list = [[8, 8, 0, 0], [0, 0, 4, 4], [0, 0, 4, 4]]

# ╔═╡ 1b6986ac-8a3c-4420-869b-42107fa31788
calculate_splitting(test_split_communities, test_split_size_list, 1)

# ╔═╡ cde909a7-c5c4-49a9-a2f8-9b61576b4954
calculate_merging(reverse(test_split_communities), reverse.(test_split_size_list), 1)

# ╔═╡ f53dd427-30eb-4dfc-af43-40dfd7a7dfd1


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Folds = "41a02a25-b8f0-4f67-bc48-60067656b558"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Folds = "~0.2.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Future", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "0264a938934447408c7f0be8985afec2a2237af4"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.11"

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

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

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

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

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

[[deps.ExternalDocstrings]]
git-tree-sha1 = "1224740fc4d07c989949e1c1b508ebd49a65a5f6"
uuid = "e189563c-0753-4f5e-ad5c-be4293c83fb4"
version = "0.1.1"

[[deps.Folds]]
deps = ["Accessors", "BangBang", "Baselet", "DefineSingletons", "Distributed", "ExternalDocstrings", "InitialValues", "MicroCollections", "Referenceables", "Requires", "Test", "ThreadedScans", "Transducers"]
git-tree-sha1 = "638109532de382a1f99b1aae1ca8b5d08515d85a"
uuid = "41a02a25-b8f0-4f67-bc48-60067656b558"
version = "0.2.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

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

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

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

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

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

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

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
# ╠═7e546c44-4e50-412f-9144-bb2056d749ff
# ╠═91d589d0-a609-11ec-2dbe-bb52470686eb
# ╠═5abe5bbb-5ed5-4ff8-8ac6-4b253e4c82d1
# ╠═a93406c3-db78-403d-bc7f-3d6de25ce608
# ╠═79511d3a-1e2c-4aea-9889-1c5c8844746e
# ╠═1eb5bc1b-52d5-4482-a2cf-e83b044b15f5
# ╠═926c5c37-b87b-480e-b042-f89b7c4b456d
# ╠═10041156-7aba-4fb9-a172-e7fe215fba8a
# ╠═783d617d-113e-4ec9-a144-6eb060738f3d
# ╠═5be87169-43f1-4ddf-9c5c-9af5d4504804
# ╠═c8a96dcb-65ae-4562-a05d-a6d286d626cc
# ╠═d169f9fd-fd0b-4209-967e-fd834439cb28
# ╠═bdc427d8-7b17-4935-9995-2dd7e1937658
# ╠═8d11583f-eccf-45e0-8624-5f5158faf632
# ╠═629914f9-6411-417f-9eb8-b812a47551a1
# ╠═b92c7770-e5f0-4818-8b68-ee685e50123d
# ╠═85ef9af4-4397-47ca-8ecf-5d1aa9d2193d
# ╠═cada7919-cfc6-46ae-a438-025dbef90b2a
# ╠═04037256-85ac-4064-96ed-c9183ff34b1f
# ╠═7c10c2f1-158a-4158-8467-a8be9574460a
# ╠═ee187301-524f-4452-be96-bd94e68aa4f9
# ╠═1c3e0b4a-0447-4298-99c3-d5ef57e6b16a
# ╠═1724765f-5ba7-41b3-90af-70fc890e0053
# ╠═1b6986ac-8a3c-4420-869b-42107fa31788
# ╠═cde909a7-c5c4-49a9-a2f8-9b61576b4954
# ╠═f53dd427-30eb-4dfc-af43-40dfd7a7dfd1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
