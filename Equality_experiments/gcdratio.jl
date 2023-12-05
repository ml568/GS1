function gcdratio(path, seed, results_df, X, y, f, g, fStar, Li, n, d, maxIter, trackIter)
    w0 = zeros(d, 1);
    # Greedy coordinate descent Li (approximate)
    @printf("Greedy selection Li (approximate)\n")
    if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLiApx.jld")
        greedyLiApx = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLiApx.jld")
        if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLiApxfstar.jld")
            greedyLiApxfstar = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLiApxfstar.jld")
        end
    else
        w = copy(w0)
        greedyLiApx = []
        greedyLiApxfstar = []
        push!(greedyLiApx, (f(w)))
        push!(greedyLiApxfstar, (f(w)- fStar))
        for i in 1:maxIter
            grad = g(w)
            mu = sum(grad) / length(grad)
            j1 = argmax((grad .- mu) ./ sqrt.(Li))
            j2 = argmin((grad .- mu) ./ sqrt.(Li))
            dir = -(grad[j1] - grad[j2]) / (Li[j1] + Li[j2])
            w[j1] += dir
            w[j2] -= dir
            if mod(i, trackIter) == 0
                @printf("Iteration %d, Function = %f\n", i, (f(w)))
                push!(greedyLiApx, (f(w)))
                push!(greedyLiApxfstar, (f(w)- fStar))
            end
        end
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLiApx.jld"; greedyLiApx)
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLiApxfstar.jld"; greedyLiApxfstar)
        println(first(results_df, 5))
        greedyLiApx = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLiApx.jld")
        greedyLiApxfstar = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLiApxfstar.jld")
    end
    results_df = hcat(results_df, DataFrame(greedyLiApx))
    results_df = hcat(results_df, DataFrame(greedyLiApxfstar))
    return results_df
end



# # Greedy coordinate descent Li (approximate 2)
# @printf("Greedy selection Li (approximate 2)\n")
# if isfile(path * "_seed" * string(seed) * "greedyLiApx2.jld")
#     greedyLiApx2 = load(path * "_seed" * string(seed) * "greedyLiApx2.jld")
# else
#     w = copy(w0)
#     greedyLiApx2 = []
#     push!(greedyLiApx2, (f(w)))
#     for i in 1:maxIter
#         grad = g(w)
#         mu = sum(grad) / length(grad)
#         if mod(i, 2) == 0
#             j1 = argmax((grad .- mu) ./ sqrt.(Li))
#             j2 = argmax(((grad .- grad[j1]) .^ 2) ./ (Li .+ Li[j1]))
#         else
#             j1 = argmin((grad .- mu) ./ sqrt.(Li))
#             j2 = argmax(((grad .- grad[j1]) .^ 2) ./ (Li .+ Li[j1]))
#         end
#         dir = -(grad[j1] - grad[j2]) / (Li[j1] + Li[j2])
#         w[j1] += dir
#         w[j2] -= dir
#         if mod(i, trackIter) == 0
#             @printf("Iteration %d, Function = %f\n", i, (f(w)))
#             push!(greedyLiApx2, (f(w)))
#         end
#     end
#     jldsave(path * "_seed" * string(seed) * "greedyLiApx2.jld"; greedyLiApx2)
#     println(first(results_df, 5))
#     greedyLiApx2 = load(path * "_seed" * string(seed) * "greedyLiApx2.jld")
# end
# results_df = hcat(results_df, DataFrame(greedyLiApx2))
