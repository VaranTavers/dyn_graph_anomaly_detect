begin
	using Graphs
	using Distributed
	using CSV
	using DataFrames
	using SimpleWeightedGraphs
	using Folds
end

include("implementation_aco.jl")

struct ACOKSettings 
	acos # :: ACOSettings
	k:: Real
	# There are situations when the ACO algorithm is unable to create the k subgraph
	# There is two options there:
	# 	false - skip the solution (faster, but might give worse answers, this is recommended if you have points with no neighbours)
	# 	true  - regenerate solution until a possible one is created (slower, but might give better answers)
	force_every_solution:: Bool
	ACOKSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, k, f) = new(ACOSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, (_, _) -> 1.0, (_, _) -> 1.0), k, f)
	ACOKSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, e_f, c_s, k, f) = new(ACOSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, e_f, c_s), k, f)
	ACOKSettings(acos, k, f) = new(acos, k, f)
end


# Calculates the probabilities of choosing edges to add to the solution.
function calculate_probabilities(inner::ACOInner, i, vars::ACOSettings)
	graph, n, η, τ = spread(inner)

	# graph.weights[i,j] * 
	p = [ (τ[i, j]^vars.α * η[i, j]^vars.β) for j in 1:n]

	if maximum(p) == 0
		p[i] = 1
	end
	s_p = sum(p)

	p ./= s_p

	p
end

function generate_s(inner::ACOInner, vars::ACOKSettings, i)
	points = zeros(Int64, vars.k)
	points[1] = i
	for i in 2:vars.k
		points[i] = sample(calculate_probabilities(inner, points[i-1], vars.acos))
		if points[i] == points[i - 1]
			if vars.force_every_solution
				return generate_s(inner, vars)
			else
				return
			end
		end
	end

	points
end

function choose_iteration_best(inner::ACOInner, settings::ACOSettings, iterations)
	iterations = filter(x -> x != nothing, iterations)
	points = Folds.map(x -> settings.eval_f(inner.graph, settings.compute_solution(inner.graph, x)), iterations)
	index = argmax(points)
	(iterations[index], points[index])
end

begin
	fst((a, _)) = a
	snd((_, b)) = b
end

function calculate_η_ij(graph, i, j, m)
	if graph.weights[i, j] == 0
		return 0;
	end
	
	count(graph.weights[:, j] .> 0 .&& graph.weights[:, i] .> 0) / nv(graph) + count(graph.weights[:, j] .> 0 .&& graph.weights[:, i] .== 0) / nv(graph) / 4
end

function calculate_η(graph)
	n = nv(graph)

	m = minimum(graph.weights)
	if m >= 0
		m = 0
	end
	η = [ calculate_η_ij(graph, i, j, m) for i in 1:n, j in 1:n]

	η
end

function calculate_denseness(_g, c)
	length(c)
end

function compute_solution(graph, s_o)
	s = collect(unique(s_o))
	edges_else = []
	for i in 1:length(s)
		for j in i:length(s)
			if graph.weights[s[i], s[j]] != 0
				if s[i] < s[j]
					push!(edges_else, (s[i], s[j]))
				else
					push!(edges_else, (s[j], s[i]))
				end
			end
		end
	end
	edges_else
end

function ACOK(graph, vars::ACOKSettings, η, τ)
	#Set parameters and initialize pheromone traits.
	n, _ = size(η)
	inner = ACOInner(graph, n, η, τ)
	
	@assert nv(graph) >= vars.k
	sgb = [i for i in 1:n]
	sgb_val = -1000
	τ_max = vars.acos.starting_pheromone_ammount
	τ_min = 0
	
	# While termination condition not met
	for i in 1:vars.acos.max_number_of_iterations
		# Construct new solution s according to Eq. 2

		if i < vars.acos.max_number_of_iterations - 3
			S = Folds.map(x -> generate_s(inner, vars, rand(1:inner.n)), zeros(vars.acos.number_of_ants))
		else
			S = Folds.map(x -> generate_s(inner, vars, x), 1:inner.n)
		end

		if length(filter(x -> x != nothing, S)) > 0
			# Update iteration best
			(sib, sib_val) = choose_iteration_best(inner, vars.acos, S)
			if sib_val > sgb_val
				sgb_val = sib_val
				sgb = sib
				
				# Compute pheromone trail limits
				τ_max = sgb_val / (1 - vars.acos.ρ)
				τ_min = vars.acos.ϵ * τ_max
			end
			# Update pheromone trails
			# TODO: test with matrix sum
			τ .*= (1 - vars.acos.ρ)
			for (a, b) in zip(sib, sib[2:end])
				τ[a, b] += sib_val
				τ[b, a] += sib_val
	
			end
		end
		τ = min.(τ, τ_max)
		τ = max.(τ, τ_min)

	end
	
	vars.acos.compute_solution(inner.graph, sgb), τ
end

function ACOK(graph, vars::ACOKSettings, η)
	n, _ = size(η)
	τ = ones(n, n) .* vars.acos.starting_pheromone_ammount
	r, _ = ACOK(graph, vars, η, τ)

	r
end


function ACOK_get_pheromone(graph, vars::ACOKSettings, η)
	n, _ = size(η)
	τ = ones(n, n) .* vars.acos.starting_pheromone_ammount
	ACOK(graph, vars, η, τ)
end

function copy_replace_funcs(vars_base::ACOKSettings, eval_f, c_s)
	ACOKSettings(
		ACOSettings(
			vars_base.acos.α,
			vars_base.acos.β,
			vars_base.acos.number_of_ants,
			vars_base.acos.ρ,
			vars_base.acos.ϵ,
			vars_base.acos.max_number_of_iterations,
			vars_base.acos.starting_pheromone_ammount,
			eval_f,
			c_s
		),
		vars_base.k,
		vars_base.force_every_solution,
	)
end

function DensestACOK(graph, vars_base::ACOKSettings, τ)
	η = calculate_η(graph)

	vars = copy_replace_funcs(vars_base, calculate_denseness, compute_solution)

	ACOK(graph, vars, η, τ)
end

function DensestACOK(graph, vars_base::ACOKSettings)
	η = calculate_η(graph)

	vars = copy_replace_funcs(vars_base, calculate_denseness, compute_solution)

	ACOK(graph, vars, η)
end

function DensestACOK_get_pheromone(graph, vars_base::ACOKSettings)
	η = calculate_η(graph)

	vars = copy_replace_funcs(vars_base, calculate_denseness, compute_solution)
	
	ACOK_get_pheromone(graph, vars, η)
end

function solution_to_community(graph, solution)
	n = nv(graph)

	f = fst.(solution)
	s = snd.(solution)
	append!(f, s)

	[ findfirst(x->x==i, f) != nothing ? 1 : 0 for i in 1:n]
end
