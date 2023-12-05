function gsq(path, seed, results_df,  X, y, f, g, fStar, Li, n, d, maxIter, trackIter)
    w0 = zeros(d, 1)
    # Greedy coordinate descent Li (exact)
    @printf("Greedy selection Li (exact)\n")
    if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLi.jld")
        greedyLi = load(path * "_seed" * string(seed) *  "iter" * string(maxIter)*"greedyLi.jld")
        if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLifstar.jld")
            greedyLifstar = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLifstar.jld")
        end
    else
        w = copy(w0)
        greedyLi = []
        greedyLifstar = []
        push!(greedyLi, (f(w)))
        push!(greedyLifstar, (f(w) - fStar))
        for i in 1:maxIter
            grad = g(w)
            maxDelta = 0
            maxj1 = 0
            maxj2 = 0
            for j1 in 1:d
                for j2 in 1:d
                    delta = (grad[j1] - grad[j2])^2 / (Li[j1] + Li[j2])
                    if delta > maxDelta
                        maxDelta = delta
                        maxj1 = j1
                        maxj2 = j2
                    end
                end
            end
            j1 = maxj1
            j2 = maxj2
            dir = -(grad[j1] - grad[j2]) / (Li[j1] + Li[j2])
            w[j1] += dir
            w[j2] -= dir
            if mod(i, trackIter) == 0
                @printf("Iteration %d, Function = %f\n", i, (f(w)))
                push!(greedyLi, (f(w)))
                push!(greedyLifstar, (f(w) - fStar))
            end
        end
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLi.jld"; greedyLi)
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLifstar.jld"; greedyLifstar)
        println(first(results_df, 5))
        greedyLi = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLi.jld")
        greedyLifstar = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLifstar.jld")
    end
    results_df = hcat(results_df, DataFrame(greedyLi))
    results_df = hcat(results_df, DataFrame(greedyLifstar))
    return results_df
end