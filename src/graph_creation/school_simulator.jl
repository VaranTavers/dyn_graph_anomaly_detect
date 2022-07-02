### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 6611c9c0-a84e-11ec-072c-0b286e94fbeb
begin
	using Graphs
	using SimpleWeightedGraphs
	using PlutoUI
	using Random
	using GraphPlot
	using Folds
	using Colors
	using CSV
	using DataFrames
end

# ╔═╡ 30b07463-2a7f-4ace-91d6-73d1d8be31a9
md"""
Number of people:
$(@bind number_of_people Scrubbable(0:50:1000))
"""

# ╔═╡ 62aea3fe-8050-4817-9458-af98aeae88b6
md"""
Number of classes:
$(@bind number_of_classes Scrubbable(2:15))
"""

# ╔═╡ 4b034fbb-32d2-4685-ade2-da91b86c0680
md"""
Number of friend groups per class:
$(@bind number_of_friend_groups Scrubbable(2:10))
"""

# ╔═╡ b9ade87f-fd67-4996-84f2-ca2401081e4f
md""" Number of places in recess:
$(@bind number_of_places Scrubbable(1:10))
"""

# ╔═╡ c7608d30-a159-4464-a1b0-4705466149f5
md"""
Number of teachers:
$(@bind number_of_teachers Scrubbable(number_of_classes: number_of_classes + 10))
"""

# ╔═╡ d00b3544-3774-4126-abc8-62e534d8a478
md"""
Number of classes/day:
$(@bind number_of_classes_per_day Scrubbable(3:8))
"""

# ╔═╡ e3925cfa-9f1d-4f82-a27b-5e8129b2829b
people_classes = rand(collect(1:number_of_classes), number_of_people)

# ╔═╡ cbffd573-71dc-4f2b-80ba-ba1240f1d56e
people_groups = map(x -> rand(collect(((x-1) * number_of_friend_groups + 1):x*number_of_friend_groups)) , people_classes)

# ╔═╡ 6cdb2cba-9925-4c37-a573-540a014941a4
groups_places = rand(collect(1:number_of_places), number_of_friend_groups * number_of_classes)

# ╔═╡ 8f5e857c-a51d-4d15-b2b5-bf00f686b1c9
# We will pemute this vector to get a timetable for each class
teacher_classes = vcat(collect(1:number_of_classes), zeros(Int64, number_of_teachers - number_of_classes)) 

# ╔═╡ d2c3d71a-1548-417e-9264-acb852299443
md"""
Ratio of interaction with other class, during class per interval (x / 1000):
$(@bind p_class_inter Scrubbable(0:200))
"""

# ╔═╡ 39b4915b-57f8-4655-adc6-b3a3d3ae068d
md"""
Ratio of interaction with other group, during break per interval (x / 1000):
$(@bind p_group_inter Scrubbable(0:200))
"""

# ╔═╡ a124a4c8-398b-4f6d-aa83-aaee916b7760
md"""
Time intervals in minutes:
$(@bind δ_t Scrubbable(1:60))
"""

# ╔═╡ 242aa867-9430-4920-9727-301194d134fa
md"""
Verbose logging $(@bind verbose CheckBox())
"""

# ╔═╡ 42688150-a3e3-4d2a-8c5e-c6620b4c43b0
timetable = [shuffle(teacher_classes) for _ in 1:number_of_classes_per_day]

# ╔═╡ 2a7b010e-603c-4d3b-a63b-76cfe44b5055
function log!(log, string, i)
	if verbose || i == 1
		push!(log, string)
	end
end

# ╔═╡ 42b340a2-1640-4766-9783-5cf40832bf53
function calculate_class_time_person(logs, time, m)
	class = people_classes[m]
	if rand() < p_class_inter / 1000
		class = rand(1:number_of_classes)
		log!(logs, "Classtime switch at $(time) , person: $(m), $(people_classes[m]) -> $(class)", 1)
	end

	class
end

# ╔═╡ 7823c46b-3142-4606-a286-01a6defaf1b3
function calculate_class_time_people(logs, time)
	[ calculate_class_time_person(logs, time, m) for m in 1:number_of_people]
end

# ╔═╡ 8f47f326-14bb-4e35-a272-4ef23c35ff92
function calculate_break_time_person(logs, time, m)
	group = people_groups[m]
	if rand() < p_group_inter / 1000
		group = rand(1:number_of_friend_groups)
		log!(logs, "Breaktime switch at $(time) , person: $(m), $(people_groups[m]) -> $(group)", 1)
	end

	group
end

# ╔═╡ 94063385-f505-4b74-a915-5a5e26a2274c
function calculate_break_time_people(logs, time)
	[ calculate_break_time_person(logs, time, m) for m in 1:number_of_people]
end

# ╔═╡ 5bf8ff8f-c3bb-43fd-9298-b15c985ade12
function get_place(groups)
	map(x -> groups_places[x], groups)
end

# ╔═╡ 1512550e-144a-4753-87bf-d0f453ce0915
function calculate_class_time_all(logs, i)
	vcat(calculate_class_time_people(logs, i), timetable[Int64(ceil(i / 60))])
end

# ╔═╡ b3ab9aba-6c15-4e58-926c-f00d2950d642
function calculate_break_time_all(logs, i)
	vcat(calculate_break_time_people(logs, i), [9999 for i in 1:number_of_teachers])
end

# ╔═╡ bc4e4d34-cd8d-4751-b97e-d0e991a5914c
begin
	n = number_of_people + number_of_teachers
	logs = []
	
	people_place_table = [ i % 60 < 50 ? calculate_class_time_all(logs, i) : calculate_break_time_all(logs, i) for i in 1:(number_of_classes_per_day * 60)]
end

# ╔═╡ 519686e3-2c3c-46b3-97b0-20077a42ead9
function create_interval(start_time::Integer, time_interval::Integer)
	n = number_of_people + number_of_teachers
	g = SimpleWeightedGraph(n)

	mat = zeros(n, n)
	
	for t in start_time:(start_time + time_interval - 1)
		for m in 1:n
			same_class = people_place_table[t] .== people_place_table[t][m]
			mat[m, same_class] .+= 1
		end
	end

	

	for i in 1:n
		for j in (i+1):n
			if mat[i,j] + mat[j, i] > 0
				add_edge!(g, i, j, (mat[i,j] + mat[j, i]) / (2 * time_interval))
			end
		end
	end

	g
end

# ╔═╡ 5b0f0db9-8ac1-41f2-8dec-7e5d3ba5a9bc
graphs = Folds.map(i -> create_interval(Int64((i - 1) * δ_t + 1), δ_t), 1:(ceil(number_of_classes_per_day * 60 / δ_t - 1)))

# ╔═╡ 29fee730-866b-456a-b6c7-0e7c30c2152c
@bind g_i Scrubbable(1:length(graphs))

# ╔═╡ 0639a2cf-3181-407e-bac5-4969d679e851
begin
	m = number_of_people + number_of_teachers
	colors = distinguishable_colors(3)[2:end]
	node_colors = [colors[i <= number_of_people ? 1 : 2] for i in 1:m]
	nodelabel=collect(1:m)
	gplot(graphs[g_i], nodesize=5, nodelabel=nodelabel, nodefillc=node_colors, layout=(args...)->spring_layout(args...; C=5))
end

# ╔═╡ 265741e9-2057-49d3-9da3-792471414ba0
md"""
Név:
$(@bind name TextField())
"""

# ╔═╡ 7b5b8370-bd48-4c5d-8ae1-20e694ae9591
md"""
$(@bind go CheckBox()) Create
"""

# ╔═╡ 3abf3570-4de2-457b-9197-b15c3683c818
begin
	if go
		for (i, g) in enumerate(graphs)
			savegraph("dynamic_graphs/$(name)$(i).lgz", g)
		end
		for (i, g) in enumerate(graphs)
			CSV.write("dynamic_communities/$(name)$(i).dat", Tables.table(people_place_table[i]'))
		end
		open("logs/$(name).txt","w") do io
				println(io, "Number of people $(number_of_people)")
				println(io, "Number of classes $(number_of_classes)")
				println(io, "Number of friend groups $(number_of_friend_groups)")
				println(io, "Number of teachers $(number_of_teachers)")
				println(io, "Number of classes/day $(number_of_classes_per_day)")
				println(io, "Ratio of interaction with other class, during class per 5 minutes (x / 1000): $(p_class_inter)")
				println(io, "Ratio of interaction with other group, during break per minute (x / 1000): $(p_group_inter)")
				println(io, "Time intervals in minutes: $(δ_t)")
				println(io, "People-Class vector");
				println(io, people_classes);
				println(io, "People-Group vector");
				println(io, people_groups);
				println(io, "Group-place vector");
				println(io, groups_places)
				println(io, "Teacher timetable");
				println(io, timetable);
   				println(io,logs)
			end
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Folds = "41a02a25-b8f0-4f67-bc48-60067656b558"
GraphPlot = "a2cc645c-3eea-5389-862e-a155d0052231"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622"

[compat]
CSV = "~0.10.3"
Colors = "~0.12.8"
DataFrames = "~1.3.4"
Folds = "~0.2.7"
GraphPlot = "~0.5.0"
Graphs = "~1.6.0"
PlutoUI = "~0.7.23"
SimpleWeightedGraphs = "~1.2.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

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
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "9310d9495c1eb2e4fa1955dd478660e2ecab1fbb"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.3"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "32a2b8af383f11cbb65803883837a149d10dfe8a"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.10.12"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "9a2695195199f4f20b94898c8a8ac72609e165a4"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.3"

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
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

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
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

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

[[deps.GraphPlot]]
deps = ["ArnoldiMethod", "ColorTypes", "Colors", "Compose", "DelimitedFiles", "Graphs", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "5e51d9d9134ebcfc556b82428521fe92f709e512"
uuid = "a2cc645c-3eea-5389-862e-a155d0052231"
version = "0.5.0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "57c021de207e234108a6f1454003120a1bf350c4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.6.0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

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

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

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

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

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
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "5152abbdab6488d5eec6a01029ca6697dff4ec8f"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.23"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

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
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

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
git-tree-sha1 = "6976fab022fea2ffea3d945159317556e5dad87c"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.2"

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

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

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

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

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
# ╠═6611c9c0-a84e-11ec-072c-0b286e94fbeb
# ╠═30b07463-2a7f-4ace-91d6-73d1d8be31a9
# ╟─62aea3fe-8050-4817-9458-af98aeae88b6
# ╟─4b034fbb-32d2-4685-ade2-da91b86c0680
# ╟─b9ade87f-fd67-4996-84f2-ca2401081e4f
# ╟─c7608d30-a159-4464-a1b0-4705466149f5
# ╟─d00b3544-3774-4126-abc8-62e534d8a478
# ╠═e3925cfa-9f1d-4f82-a27b-5e8129b2829b
# ╠═cbffd573-71dc-4f2b-80ba-ba1240f1d56e
# ╠═6cdb2cba-9925-4c37-a573-540a014941a4
# ╠═8f5e857c-a51d-4d15-b2b5-bf00f686b1c9
# ╟─d2c3d71a-1548-417e-9264-acb852299443
# ╠═39b4915b-57f8-4655-adc6-b3a3d3ae068d
# ╟─a124a4c8-398b-4f6d-aa83-aaee916b7760
# ╟─242aa867-9430-4920-9727-301194d134fa
# ╠═42688150-a3e3-4d2a-8c5e-c6620b4c43b0
# ╟─2a7b010e-603c-4d3b-a63b-76cfe44b5055
# ╠═42b340a2-1640-4766-9783-5cf40832bf53
# ╠═7823c46b-3142-4606-a286-01a6defaf1b3
# ╠═8f47f326-14bb-4e35-a272-4ef23c35ff92
# ╠═94063385-f505-4b74-a915-5a5e26a2274c
# ╠═5bf8ff8f-c3bb-43fd-9298-b15c985ade12
# ╠═1512550e-144a-4753-87bf-d0f453ce0915
# ╠═b3ab9aba-6c15-4e58-926c-f00d2950d642
# ╠═bc4e4d34-cd8d-4751-b97e-d0e991a5914c
# ╠═519686e3-2c3c-46b3-97b0-20077a42ead9
# ╠═5b0f0db9-8ac1-41f2-8dec-7e5d3ba5a9bc
# ╠═29fee730-866b-456a-b6c7-0e7c30c2152c
# ╠═0639a2cf-3181-407e-bac5-4969d679e851
# ╠═265741e9-2057-49d3-9da3-792471414ba0
# ╠═7b5b8370-bd48-4c5d-8ae1-20e694ae9591
# ╠═3abf3570-4de2-457b-9197-b15c3683c818
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
