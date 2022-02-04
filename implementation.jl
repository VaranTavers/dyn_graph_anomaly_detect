### A Pluto.jl notebook ###
# v0.17.4

using Markdown
using InteractiveUtils

# ╔═╡ 57912b3a-83ae-11ec-0fbf-9da8ce954fe1
begin
	using Graphs
	using SimpleWeightedGraphs
end

# ╔═╡ 57912b1c-83ae-11ec-2a63-57bf99a1eb84
html"""
	<div>Happy holiday! Remember to take care of yourself and your loved ones!</div>
<div id="snow"></div>
<style>
	body:not(.disable_ui):not(.more-specificity) {
        background-color:#e9ecff;
    }
	pluto-output{
		border-radius: 0px 8px 0px 0px;
        background-color:#e9ecff;
	}
	#snow {
        position: fixed;
    	top: 0;
    	left: 0;
    	right: 0;
    	bottom: 0;
    	pointer-events: none;
    	z-index: 1000;
	}
</style>
<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
        setTimeout(() => window.particlesJS("snow", {
            "particles": {
                "number": {
                    "value": 70,
                    "density": {
                        "enable": true,
                        "value_area": 800
                    }
                },
                "color": {
                    "value": "#ffffff"
                },
                "opacity": {
                    "value": 0.7,
                    "random": false,
                    "anim": {
                        "enable": false
                    }
                },
                "size": {
                    "value": 5,
                    "random": true,
                    "anim": {
                        "enable": false
                    }
                },
                "line_linked": {
                    "enable": false
                },
                "move": {
                    "enable": true,
                    "speed": 5,
                    "direction": "bottom",
                    "random": true,
                    "straight": false,
                    "out_mode": "out",
                    "bounce": false,
                    "attract": {
                        "enable": true,
                        "rotateX": 300,
                        "rotateY": 1200
                    }
                }
            },
            "interactivity": {
                "events": {
                    "onhover": {
                        "enable": false
                    },
                    "onclick": {
                        "enable": false
                    },
                    "resize": false
                }
            },
            "retina_detect": true
        }), 3000);
	</script>
"""


# ╔═╡ fefdd2c0-02a7-4db3-b868-f40c98373e1f
# Kronecker delta function
function δ(i, j)
	if i == j
		return 1
	end
	0
end

# ╔═╡ c88acc56-32a0-4209-aa92-eec60e1bbbe7
md"""
$A_{ij} - \frac{k_i * k_j}{2m}$
"""

# ╔═╡ bf2fe679-e748-4117-9425-c4285f0d729e
function calculate_inner(graph, i, j)
	m = ne(graph)
	A_ij = has_edge(graph, i, j) ? 1 : 0
	k_i = length(all_neighbors(graph, i))
	k_j = length(all_neighbors(graph, j))

	A_ij - (k_i * k_j) / 2m
end

# ╔═╡ 74f178c1-4ecd-4ecf-8a08-ce3a4dbdb766
function calculate_modularity(graph, c)
	m = ne(graph)

	s = sum(Iterators.flatten([
		[ calculate_inner(graph, i, j) * δ(c[i], c[j])
		for j in i:nv(graph)] for i in 1:nv(graph)]))

	1 / (2 * m) * s
end

# ╔═╡ 4ea094e2-d266-4b92-9361-49511d8b7b0b
begin
	# Building the graph that is presented in the paper (Fig. 1)
	g = SimpleWeightedGraph(12)
	
	add_edge!(g, 1, 2, 1)
	add_edge!(g, 1, 3, 1)
	add_edge!(g, 1, 4, 1)
	add_edge!(g, 2, 3, 1)
	add_edge!(g, 2, 4, 1)
	add_edge!(g, 3, 4, 1)
	add_edge!(g, 3, 5, 1)
	add_edge!(g, 5, 6, 1)
	add_edge!(g, 5, 7, 1)
	add_edge!(g, 5, 8, 1)
	add_edge!(g, 5, 9, 1)
	add_edge!(g, 6, 7, 1)
	add_edge!(g, 6, 8, 1)
	add_edge!(g, 6, 9, 1)
	add_edge!(g, 7, 8, 1)
	add_edge!(g, 7, 9, 1)
	add_edge!(g, 8, 9, 1)
	add_edge!(g, 7, 10, 1)
	add_edge!(g, 10, 11, 1)
	add_edge!(g, 10, 12, 1)
	add_edge!(g, 11, 12, 1)

	g
end

# ╔═╡ 9cc44d49-972a-4590-8301-1f7c443ee731
begin
	# Calculating Q for the afforementioned graph. The result is 0.102, which means there are more edges in communities than what we'd expect in a random graph.
	c = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
	calculate_modularity(g, c)
end

# ╔═╡ ca6fe67f-4f76-4cd2-a310-97211d61cef9
function sq(x)
	x * x
end

# ╔═╡ 8b2e3cd3-fb5d-4a81-8cf4-27b956088bab
md"""
Global settings for the Ant colony optimalization algorithm. 
"""

# ╔═╡ 8b130d9c-5712-45cb-8486-d07a1a98a894
begin
	α = 1
	β = 2
	number_of_ants = 30
	ρ = 0.8
	ϵ = 0.005
	max_number_of_iterations = 100
end

# ╔═╡ e66b3fe1-7979-4811-a349-4b027e112310
md"""
$C(i,j) = \frac{\sum_{v_t \in V}{(A_{il} - \mu_i)(A_{jl} - \mu_j)}}{n\sigma_i\sigma_j}$
"""

# ╔═╡ 0bbaaf25-4633-4a82-859e-db81068d680a
function pearson_corelation(graph::SimpleWeightedGraph{Int64, Float64}, i, j)
	n = nv(graph)
	
	μ_i = sum(map(x -> graph.weights[i, x], 1:n)) / nv(graph)
	μ_j = sum(map(x -> graph.weights[j, x], 1:n)) / nv(graph)
	σ_i = sqrt(sum(map(x -> sq(graph.weights[i, x] - μ_i), 1:n)) / n)
	σ_j = sqrt(sum(map(x -> sq(graph.weights[j, x] - μ_j), 1:n)) / n)

	numerator = sum(
		[(graph.weights[i, x] - μ_i) * (graph.weights[j, x] - μ_j) for x in 1:n]
	)

	numerator / (n * σ_i * σ_j)
end

# ╔═╡ 39bdd460-58de-4bcc-8237-12586b74a11b
function logistic(x)
	1 / (1 + exp(-x))
end

# ╔═╡ adaaeb50-f117-45ff-934b-890be5e972fe
# Calculates the probabilities of choosing edges to add to the solution.
function calculate_probabilities(graph, η, τ, i)
	n = nv(graph)

	p = [(graph.weights[i,j] > 0 ? τ[i, j]^α * η[i, j]^β : 0) for j in 1:n]
	
	s_p = sum(p)

	p ./= s_p

	p
end

# ╔═╡ f60bc671-65a7-4335-8388-22535751d23b
sample(weights) = findfirst(cumsum(weights) .> rand())

# ╔═╡ 467549ca-519a-4a84-98e7-9e78d93342a2
# Constructs a new solution
function generate_s(graph, η, τ)
	n = nv(graph)
	s = [(0, 0) for i in 1:n]

	for i in 1:n
		res = sample(calculate_probabilities(graph, η, τ, i))
		s[i] = (i, res)
	end

	s
end

# ╔═╡ 56269167-b380-4940-8278-adaa01356650
# Built for bidirectional edges
# Transforms the edge representation from generate_s to a community vector.
function compute_solution(n, η, τ, edges)
	@show edges
	tmp_g = SimpleWeightedGraph(n)
	for (a, b) in edges
		add_edge!(tmp_g, a, b)
	end
	
	s = [0 for i in 1:n]
	start = 1
	clust = 1
	while start <= n
		while start <= n && s[start] != 0
			start += 1
		end
		if start > n
			break
		end
		s[start] = clust
		for (j, v) in enumerate(dfs_parents(tmp_g, start))
			if v > 0
				s[j] = clust
			end
		end
		clust += 1
	end
	
	s
end

# ╔═╡ aa8314fd-ab17-4e23-9e83-dc79a6f69209
function choose_iteration_best(graph, η, τ, iterations)
	n = nv(graph)
	points = [calculate_modularity(graph, compute_solution(n, η, τ, x)) for x in iterations]
	index = argmax(points)

	(iterations[index], points[index])
end

# ╔═╡ 8167196c-4a45-45f0-b55b-26b69f27904b
function ACO(graph)
	#Set parameters and initialize pheromone traits.
	n = nv(graph)
	
	η = [logistic(pearson_corelation(graph, i, j)) * (1 - δ(i, j)) for i in 1:n, j in 1:n]
	τ = ones(n, n) # TODO set to relatively high
	sgb = [i for i in 1:n]
	sgb_val = -1000
	
	# While termination condition not met
	for i in 1:max_number_of_iterations
		S = []
		for j in 1:number_of_ants
			# Construct new solution s according to Eq. 2
			append!(S, [generate_s(graph, η, τ)])
		end
		
		# Update iteration best
		(sib, sib_val) = choose_iteration_best(graph, η, τ, S)
		if sib_val > sgb_val
			sgb_val = sib_val
			sgb = sib
		end
		
		# Compute pheromone trail limits
		τ_max = calculate_modularity(graph, sgb) / (1 - ρ)
		τ_min = ϵ * τ_max

		# Update pheromone trails
		τ .*= ρ
		for (a, b) in sib
			τ[a, b] += sib_val
			τ[b, a] += sib_val
		end
		τ = min.(τ, τ_max)
		τ = max.(τ, τ_min)

	end
	compute_solution(n, η, τ,sgb)
end

# ╔═╡ 39d3be8a-402e-4151-957b-a1832cca6ded
ACO(g)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622"

[compat]
Graphs = "~1.5.1"
SimpleWeightedGraphs = "~1.2.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "d727758173afef0af878b29ac364a0eca299fc6b"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.5.1"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

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

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "2884859916598f974858ff01df7dfc6c708dd895"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

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
# ╟─57912b1c-83ae-11ec-2a63-57bf99a1eb84
# ╠═57912b3a-83ae-11ec-0fbf-9da8ce954fe1
# ╠═fefdd2c0-02a7-4db3-b868-f40c98373e1f
# ╟─c88acc56-32a0-4209-aa92-eec60e1bbbe7
# ╠═bf2fe679-e748-4117-9425-c4285f0d729e
# ╠═74f178c1-4ecd-4ecf-8a08-ce3a4dbdb766
# ╠═4ea094e2-d266-4b92-9361-49511d8b7b0b
# ╠═9cc44d49-972a-4590-8301-1f7c443ee731
# ╠═ca6fe67f-4f76-4cd2-a310-97211d61cef9
# ╟─8b2e3cd3-fb5d-4a81-8cf4-27b956088bab
# ╠═8b130d9c-5712-45cb-8486-d07a1a98a894
# ╟─e66b3fe1-7979-4811-a349-4b027e112310
# ╠═0bbaaf25-4633-4a82-859e-db81068d680a
# ╠═39bdd460-58de-4bcc-8237-12586b74a11b
# ╠═adaaeb50-f117-45ff-934b-890be5e972fe
# ╠═f60bc671-65a7-4335-8388-22535751d23b
# ╠═467549ca-519a-4a84-98e7-9e78d93342a2
# ╠═56269167-b380-4940-8278-adaa01356650
# ╠═aa8314fd-ab17-4e23-9e83-dc79a6f69209
# ╠═8167196c-4a45-45f0-b55b-26b69f27904b
# ╠═39d3be8a-402e-4151-957b-a1832cca6ded
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002