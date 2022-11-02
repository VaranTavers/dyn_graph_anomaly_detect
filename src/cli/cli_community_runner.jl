begin
	using Graphs
	using SimpleWeightedGraphs
	using CSV
	using DataFrames
	using Folds
	using StatsBase
	using DSP
	using Statistics
end

include("implementation_community.jl")

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

@time begin
    for i in 1:number_of_tests
        result = CommunityACO(graph, vars)
    end
end