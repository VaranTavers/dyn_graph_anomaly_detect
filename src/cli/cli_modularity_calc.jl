begin
	using Graphs
	using SimpleWeightedGraphs
	using CSV
    using Folds
	using DataFrames
	using BenchmarkTools
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


name = ARGS[1]

communities = Matrix(CSV.read(ARGS[2], DataFrame, header=false))
communities = communities[:]
@show communities

graph = loadgraph(name, SWGFormat())

@show calculate_modularity(graph, communities)