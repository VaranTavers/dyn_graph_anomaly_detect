
begin
	using Graphs
	using Distributed
	using DataFrames
	using SimpleWeightedGraphs
	using Folds
end


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


mutable struct ACOInner
	graph
	n
	η
	τ
end

sample(weights) = findfirst(cumsum(weights) .> rand())


spread(inner::ACOInner) = inner.graph, inner.n, inner.η, inner.τ

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
	solutions = Folds.map(x -> settings.compute_solution(inner.n, x), filtered_iter)
	points = Folds.map(x -> settings.eval_f(inner.graph, x), solutions)
	index = argmax(points)
	(iterations[index], points[index])
end

function get_combined_c(c, (c_dest, c_source))
	r = copy(c)

	r[r .== c_source] .= c_dest

	r
end

function a_lt_b_not_empty(x, c)
	(a, b) = x

	if a >= b
		return false
	end

	count(c .== a) != 0 && count(c .== b) != 0
end

function get_combined_result(g, eval_f, c, x)
	comb = get_combined_c(c, x)
	val = eval_f(g, comb)
	(val, x, comb)
end

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


function number_of_communitites(c)
	com_num = map(x -> count(c .== x), 1:maximum(c))

	length(filter(x -> x > 0, com_num))
end


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


function ACO(graph, vars::ACOSettings, η; k = 0)
	n, _ = size(η)
	τ = ones(n, n) .* vars.starting_pheromone_ammount
	r, _ = ACO(graph, vars, η, τ; k=k)

	r
end


function ACO_get_pheromone(graph, vars::ACOSettings, η; k=0)
	n, _ = size(η)
	τ = ones(n, n) .* vars.starting_pheromone_ammount
	ACO(graph, vars, η, τ; k=k)
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