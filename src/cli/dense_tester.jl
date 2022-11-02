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


include("./implementation_k_dense.jl")

name_list_file = "dense_test_files_mini.csv"

number_of_tests = 50


begin
	df = CSV.read(name_list_file, DataFrame, header=true)
end


begin
	paths_zip = zip(df[!, "Folder"], df[!, "Filename"])
	paths = map(((a,b),) -> "$a/$b", paths_zip)
	
	g_s = map(name -> loadgraph("./dense/$name", SWGFormat()), paths)
end


begin
	fst((a, _)) = a
	snd((_, b)) = b
end


function good(result, real)
	prec = result / real
	
	(prec, floor(prec)) 
end


begin
	l, _ = size(df)
	is_good = zeros(Float64, (l, number_of_tests))
	precision = zeros(Float64, (l, number_of_tests))
end


for (i, g) in enumerate(g_s)
	vars = ACOSettings(
		2, # α
		2, # β
		60, # number_of_ants
		0.1, # ρ
		0.005, # ϵ
		300, # max_number_of_iterations
		300 # starting_pheromone_ammount
	)
	vars3 = ACOKSettings(
		vars,
		df[i, "K",],
		false
	)

	for j in 1:number_of_tests
		k_sub_g = DensestACOK(g, vars3)
		
		goods = good(calculate_denseness(g, k_sub_g), df[i, "Best"])

		precision[i, j] = fst(goods)
		is_good[i, j] = snd(goods)
	end
end


is_good


precision


function save_result_and_stats(filename, mat)
	rows = eachrow(mat)
	means = map(mean, rows)
	stds = map(std, rows)
	mins = map(minimum, rows)
	maxs = map(maximum, rows)

	mat_res = hcat(mat, means, stds, mins, maxs)
	

	CSV.write(filename, Tables.table(mat_res), writeheader = false)
end


save_result_and_stats("./dense_is_good.csv", is_good)


save_result_and_stats("./dense_prec.csv", precision)


df[1, "Best"]
