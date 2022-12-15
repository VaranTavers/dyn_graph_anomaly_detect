begin
	using Graphs
	using SimpleWeightedGraphs
	using CSV
	using DataFrames
	using Folds
	using StatsBase
	using DSP
	using Statistics
	using BenchmarkTools
end

struct ACOSettings
	α:: Real
	β:: Real
	number_of_ants:: Integer
	ρ:: Real
	ϵ:: Real
	max_number_of_iterations:: Integer
	starting_pheromone_ammount:: Real
	ACOSettings(α, β, n_a, ρ, ϵ, max_i, start_ph) = new(α, β, n_a, ρ, ϵ, max_i, start_ph)
end

mutable struct ACOInner
	graph
	n
	η
	τ
end

sample(weights) = findfirst(cumsum(weights) .> rand())

spread(inner::ACOInner) = inner.graph, inner.n, inner.η, inner.τ

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

# Get chosen point
function get_chosen_point(pM, i, r)
	if maximum(pM[i, :]) == 0
		return i
	end

	findfirst(pM[i, :] .> r)
end


# Constructs a new solution
function generate_s(n::Integer, pM::Matrix{Float64})
	r = rand(n)
	
	[get_chosen_point(pM, i, r[i]) for i in 1:n]
end


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


function choose_iteration_best(inner::ACOInner, settings::ACOSettings, iterations)
	filtered_iter = filter(x -> x != nothing, iterations)
	solutions = Folds.map(x -> compute_solution(inner.n, x), filtered_iter)
	points = Folds.map(x -> calculate_modularity(inner.graph, x), solutions)
	index = argmax(points)
	(iterations[index], points[index])
end

function get_η_common_neighbors(g2)
	M = ceil.(abs.(g2.weights))
	M2 = (M * M' .+ 1) / nv(g2)
	[i != j ? (g2.weights[i, j] != 0 ? M2[i, j] : 0.0) : 0.00001 for i in 1:nv(g2), j in 1:nv(g2)]
end

function CommunityACO(graph, vars::ACOSettings)
	#Set parameters and initialize pheromone traits.
	η = get_η_common_neighbors(graph)
	n, _ = size(η)
	τ = ones(n, n) .* vars.starting_pheromone_ammount
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
		τ .*= (1 - vars.ρ)
		for (a, b) in enumerate(sib)
				τ[a, b] += sib_val
		end
		τ = min.(τ, τ_max)
		τ = max.(τ, τ_min)

	end
	
	compute_solution(inner.n, sgb)
end


#### CommunityACO ####

@show Threads.nthreads()

@show ARGS

name = ARGS[1]

number_of_tests = parse(Int32, ARGS[2])

begin
	vars = ACOSettings(
		1.5, # α
		2, # β
		60, # number_of_ants
		0.7, # ρ
		0.005, # ϵ
		150, # max_number_of_iterations
		300 # starting_pheromone_ammount
	)

	apply_aco(x) = CommunityACO(x, vars)

	graph = loadgraph(name, SWGFormat())

end

result = CommunityACO(graph, vars)

# Time? Btime? benchmark?

@btime begin
    for i in 1:number_of_tests
        result = CommunityACO(graph, vars)
		@show  result
    end
end