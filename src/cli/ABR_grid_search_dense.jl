begin
	using Graphs
	using SimpleWeightedGraphs
	using CSV
	using DataFrames
	using Folds
	using Colors
	using StatsBase
	using DSP
	using Statistics
end

include("implementation_k_dense.jl")

name = "kcluster80/kcluster80_025_20_1.dat"
number_of_tests = 50
k = 20

real = 94

α_min = 0.5
α_step = 0.5
α_max = 2

## β

β_min = 0.5
β_step = 0.5
β_max = 2

## ρ

ρ_min = 0.05
ρ_step = 0.5
ρ_max = 0.4


begin
	g = loadgraph("./dense/$name", SWGFormat())
end

begin
	α_s = α_min:α_step:α_max
	β_s = β_min:β_step:β_max
	ρ_s = ρ_min:ρ_step/10:ρ_max

	α_l = length(collect(α_s))
	β_l = length(collect(β_s))
	ρ_l = length(collect(ρ_s))
end

begin
	fst((a, _)) = a
	snd((_, b)) = b
end

function good(result)
	prec = result / real
	
	(prec, floor(prec)) 
end

begin
	variations = [(α, β, ρ) for α in α_s, β in β_s, ρ in ρ_s][:]
	@show length(variations)
end

function test_with_params((α, β, ρ))
	vars = ACOSettings(
		α,
		β,
		60, # number_of_ants
		ρ,
		0.005, # ϵ
		300, # max_number_of_iterations
		300 # starting_pheromone_ammount
	)
	vars3 = ACOKSettings(
		vars,
		k,
		false
	)
	
	k_sub_g = Folds.map(_ -> DensestACOK(g, vars3), 1:number_of_tests)
	goods = Folds.map(x -> good(calculate_denseness(g, x)), k_sub_g)
	
	(fst.(goods), snd.(goods))
end

begin

	@show variations
	results = Folds.map(test_with_params, variations)
	@show results

end

begin
	is_good = fst.(results)
	precision = snd.(results)
end

function save_result_and_stats(filename, rows, variations)
	means = map(mean, rows)
	stds = map(std, rows)
	mins = map(minimum, rows)
	maxs = map(maximum, rows)

	mat = mapreduce(permutedims, vcat, rows)
	mat_res = hcat(variations, mat, means, stds, mins, maxs)
	
	CSV.write(filename, Tables.table(mat_res), writeheader = false)
end

save_result_and_stats("./denseGrid_is_good.csv", is_good, variations)

save_result_and_stats("./denseGrid_prec.csv", precision, variations)

