begin
	using Graphs
	using Distributed
	using CSV
	using DataFrames
	using SimpleWeightedGraphs
	using Folds
end

include("./implementation_aco.jl")


function logistic(x)
	1 / (1 + exp(-x))
end

function δ(i, j)
	if i == j
		return 1
	end
	0
end

function calculate_modularity_inner(graph, i, j)
	begin
	m = ne(graph)
	A_ij = has_edge(graph, i, j) ? 1 : 0
	k_i = length(all_neighbors(graph, i))
	k_j = length(all_neighbors(graph, j))

	A_ij - (k_i * k_j) / 2m
	end
end

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

function community_entropy_inner(c, i)
	n = length(c)
	nx_i = count(x -> x == i, c)

	nx_i * log(nx_i / n) / n
end

function community_entropy(c)
	- sum([community_entropy_inner(c, i) for i in 1:maximum(c)])
end

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

function mutual_information(c1, c2)
	sum([sum([mutual_information_inner(c1, c2, i, j) for j in 1:maximum(c2)]) for i in 1:maximum(c1)])
end

function normalized_mutual_information(c1, c2)
	2 * mutual_information(c1, c2) / (community_entropy(c1) + community_entropy(c2))
end

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

function same_neighbors(g, i, j)
	a = Set(all_neighbors(g, i))
	b = Set(all_neighbors(g, j))
	
	length(intersect(a, b)) / nv(g)
end

function get_η_common_neighbors(g2)
	M = ceil.(abs.(g2.weights))
	M2 = (M * M' .+ 1) / nv(g2)
	[i != j ? (g2.weights[i, j] != 0 ? M2[i, j] : 0.0) : 0.00001 for i in 1:nv(g2), j in 1:nv(g2)]
end

function CommunityACO(graph, vars_base::ACOSettings, τ; k=0)
	n = nv(graph)
	η = get_η_common_neighbors(graph)

	vars = copy_replace_funcs(vars_base, calculate_modularity, compute_solution)

	ACO(graph, vars, η, τ; k)
end

function CommunityACO(graph, vars_base::ACOSettings; k=0)
	n = nv(graph)
	η = get_η_common_neighbors(graph)

	vars = copy_replace_funcs(vars_base, calculate_modularity, compute_solution)

	ACO(graph, vars, η; k)
end

function CommunityACO_get_pheromone(graph, vars_base::ACOSettings; k=0)
	n = nv(graph)
	η = get_η_common_neighbors(graph)

	vars = copy_replace_funcs(vars_base, calculate_modularity, compute_solution)
	
	ACO_get_pheromone(graph, vars, η; k=k)
end