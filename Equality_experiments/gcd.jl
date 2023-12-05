function gcd(path, seed, results_df, X, y, f, g, fStar, Li, n, d, maxIter, trackIter)
    w0 = zeros(d, 1);
    # Greedy coordinate descent
    @printf("Greedy selection\n")
    if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter)*"greedyL.jld")
        greedyL = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyL.jld")
        if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLfstar.jld")    
            greedyLfstar = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLfstar.jld")
        end
    else
        w = copy(w0)
        greedyL = []
        greedyLfstar = []
        push!(greedyL, (f(w)))
        push!(greedyLfstar, (f(w)-fStar))
        for i in 1:maxIter
            grad = g(w)
            j1 = argmax(grad)
            j2 = argmin(grad)
            dir = -(grad[j1] - grad[j2]) / (Li[j1] + Li[j2])
            w[j1] += dir
            w[j2] -= dir
            if mod(i, trackIter) == 0
                @printf("Iteration %d, Function = %f\n", i, (f(w)))
                push!(greedyL, (f(w)))
                if (f(w)-fStar) >0
                    push!(greedyLfstar, (f(w)-fStar))
                end
            end
        end
        jldsave(path * "_seed" * string(seed) *  "iter" * string(maxIter)*"greedyL.jld"; greedyL)
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyLfstar.jld"; greedyLfstar)
        println(first(results_df, 5))
        greedyL = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "greedyL.jld")
        greedyLfstar = load(path * "_seed" * string(seed) *  "iter" * string(maxIter)*"greedyLfstar.jld")
    end
    results_df = hcat(results_df, DataFrame(greedyL))
    results_df = hcat(results_df, DataFrame(greedyLfstar))
    return results_df
end