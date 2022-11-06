begin
	using Graphs
	using Distributed
	using CSV
	using DataFrames
	using SimpleWeightedGraphs
	using Folds
end
include("./implementation_aco.jl")


begin
	fst((a, _)) = a
	snd((_, b)) = b
end

function calculate_weight_sum_of_edges(graph, ei1, ei2, g_edges)
	e1 = g_edges[ei1]
	
	if ei1 == ei2
		return abs(e1.weight)
	end
	
	e2 = g_edges[ei2]
	if e1.src == e2.src || e1.src == e2.dst || e1.dst == e2.src || e1.dst == e2.dst
		return abs(e1.weight + e2.weight)
	end

	return 0
end

function calculate_η(graph)
	n = ne(graph)
	g_edges = collect(edges(graph))

	η = [ calculate_weight_sum_of_edges(graph, i, j, g_edges) for i in 1:n, j in 1:n]

	η ./ n
end

function calculate_heaviness(graph, c)
	number_of_communities = maximum(c)
	g_edges = collect(edges(graph))
	lengths = map(x -> g_edges[x].weight, 1:ne(graph))

	weights_per_subg = [sum(lengths[c .== i]) for i in 1:number_of_communities]

	maximum(weights_per_subg)
end

function inHeaviest(graph, c)
	number_of_communities = maximum(c)
	g_edges = collect(edges(graph))
	lengths = map(x -> g_edges[x].weight, 1:ne(graph))

	weights_per_subg = [sum(lengths[c .== i]) for i in 1:number_of_communities]
	heaviest_com = argmax(weights_per_subg)

	c .== heaviest_com
end

function compute_solution(n, edges)
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

function copy_replace_funcs(vars_base, eval_f, c_s)
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

function HeaviestACO(graph, vars_base, τ)
	η = calculate_η(graph)

	vars = copy_replace_funcs(vars_base, calculate_heaviness, compute_solution)

	inHeaviest(graph, ACO(graph, vars, η, τ))
end

function HeaviestACO(graph, vars_base)
	η = calculate_η(graph)

	vars = copy_replace_funcs(vars_base, calculate_heaviness, compute_solution)

	lk = ACO(graph, vars, η)
	
	inHeaviest(graph, lk)
end

function HeaviestACO_get_pheromone(graph, vars_base)
	η = calculate_η(graph)

	vars = copy_replace_funcs(vars_base, calculate_heaviness, compute_solution)
	
	r, τ = ACO_get_pheromone(graph, vars, η, τ)

	inHeaviest(graph, r), τ
end
