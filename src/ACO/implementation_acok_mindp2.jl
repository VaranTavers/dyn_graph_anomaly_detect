begin
	using Graphs
	using Distributed
	using SimpleWeightedGraphs
	using Folds
	using Statistics
end

mutable struct ACOKInner
	graph
	n
	η
	τ
end

sample(weights) = findfirst(cumsum(weights) .> rand())

spread(inner::ACOKInner) = inner.graph, inner.n, inner.η, inner.τ

struct ACOKSettings 
	α
	β
	number_of_ants
	ρ
	ϵ
	max_number_of_iterations
	starting_pheromone_ammount
	eval_f # (graph, (result of compute_solution) )
	compute_solution # (graph, (result of generate_s))

	k:: Real
	# There are situations when the ACO algorithm is unable to create the k subgraph
	# There is two options there:
	# 	false - skip the solution (faster, but might give worse answers, this is recommended if you have points with no neighbours)
	# 	true  - regenerate solution until a possible one is created (slower, but might give better answers)
	force_every_solution:: Bool

	ACOKSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, k, f) = new(α, β, n_a, ρ, ϵ, max_i, start_ph, (_, _) -> 1.0, (_, _) -> 1.0, k, f)
	ACOKSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, e_f, c_s, k, f) = new(α, β, n_a, ρ, ϵ, max_i, start_ph, e_f, c_s, k, f)

end


# Calculates the probabilities of choosing edges to add to the solution.
function calculate_probabilities(inner::ACOKInner, row, i, vars::ACOKSettings)
	graph, n, η, τ = spread(inner)
	# graph.weights[i,j] * 
	p = [ (τ[j]^vars.α * row[j]^vars.β) for j in 1:n]
	
	if maximum(p) == 0
		p[i] = 1
	end
	s_p = sum(p)

	p ./= s_p
	p
end

function generate_s(inner::ACOKInner, vars::ACOKSettings, i)
	points = zeros(Int64, vars.k)
	points[1] = rand(1:inner.n)
	row = copy(inner.η)
	for i in 2:vars.k
		points[i] = sample(calculate_probabilities(inner, row, points[i - 1], vars))
		row[points[i]] = 0
		if points[i] == points[i - 1]
			if vars.force_every_solution
				return generate_s(inner, vars)
			end
			return
		end

	end

	points
end

function choose_iteration_best(inner::ACOKInner, settings::ACOKSettings, iterations)
	iterations = filter(x -> x != nothing, iterations)
	points = Folds.map(x -> settings.eval_f(inner.graph, settings.compute_solution(inner.graph, x)), iterations)
	index = argmax(points)
	(iterations[index], points[index])
end

begin
	fst((a, _)) = a
	snd((_, b)) = b
end

function ACOK(graph, vars::ACOKSettings, η, τ)
	#Set parameters and initialize pheromone traits.
	n = length(η)
	inner = ACOKInner(graph, n, η, τ)
	
	@assert nv(graph) >= vars.k
	sgb = [i for i in 1:n]
	sgb_val = -1000
	τ_max = vars.starting_pheromone_ammount
	τ_min = 0
	
	# While termination condition not met
	for i in 1:vars.max_number_of_iterations
		# Construct new solution s according to Eq. 2

		if i < vars.max_number_of_iterations - 3
			S = Folds.map(x -> generate_s(inner, vars, rand(1:inner.n)), zeros(vars.number_of_ants))
		else
			S = Folds.map(x -> generate_s(inner, vars, x), 1:inner.n)
		end

		if length(filter(x -> x != nothing, S)) > 0
			# Update iteration best
			(sib, sib_val) = choose_iteration_best(inner, vars, S)
			if sib_val > sgb_val
				sgb_val = sib_val
				sgb = sib
				
				# Compute pheromone trail limits
				τ_max = sgb_val / (1 - vars.ρ)
				τ_min = vars.ϵ * τ_max
			end
			# Update pheromone trails
			# TODO: test with matrix sum
			τ .*= (1 - vars.ρ)
			for a in sib
				τ[a] += sib_val
			end
		end
		τ = min.(τ, τ_max)
		τ = max.(τ, τ_min)

	end
	
	vars.compute_solution(inner.graph, sgb), τ
end

function ACOK(graph, vars::ACOKSettings, η)
	n, _ = size(η)
	τ = ones(n) .* vars.starting_pheromone_ammount
	η = mean(η, dims=1)
	r, _ = ACOK(graph, vars, η, τ)

	r
end


function ACOK_get_pheromone(graph, vars::ACOKSettings, η)
	n, _ = size(η)
	τ = ones(n) .* vars.starting_pheromone_ammount
	η = mean(η, dims=1)
	ACOK(graph, vars, η, τ)
end

function copy_replace_funcs(vars_base::ACOKSettings, eval_f, c_s)
	ACOKSettings(
		vars_base.α,
		vars_base.β,
		vars_base.number_of_ants,
		vars_base.ρ,
		vars_base.ϵ,
		vars_base.max_number_of_iterations,
		vars_base.starting_pheromone_ammount,
		eval_f,
		c_s,
		vars_base.k,
		vars_base.force_every_solution,
	)
end