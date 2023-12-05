function gs1(path, seed, results_df,  X, y, f, g, fStar, Li, n, d, maxIter, trackIter)
    w0 = zeros(d, 1)
    # Greedy coordinate descent Li (GS-1)
    @printf("Greedy selection Li GS-1\n")
    if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLigs1.jld")
        greedyLigs1 = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLigs1.jld")
        if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLigs1fstar.jld")
            greedyLigs1fstar = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLigs1fstar.jld")
        end
    else
        w = copy(w0)
        greedyLigs1 = []
        greedyLigs1fstar = []
        push!(greedyLigs1, (f(w)))
        push!(greedyLigs1fstar, (f(w) - fStar))
        for i in 1:maxIter
            grad = g(w)
            maxDelta = 0
            maxj1 = 0
            maxj2 = 0
            for j1 in 1:d
                for j2 in 1:d
                    delta = (grad[j1] - grad[j2]) / (sqrt(Li[j1]) + sqrt(Li[j2]))
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
                push!(greedyLigs1, (f(w)))
                push!(greedyLigs1fstar, (f(w) - fStar))
            end
        end
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLigs1.jld"; greedyLigs1)
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLigs1fstar.jld"; greedyLigs1fstar)
        println(first(results_df, 5))
        greedyLigs1 = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLigs1.jld")
        greedyLigs1fstar = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLigs1fstar.jld")
    end
    results_df = hcat(results_df, DataFrame(greedyLigs1))
    results_df = hcat(results_df, DataFrame(greedyLigs1fstar))
    return results_df
end