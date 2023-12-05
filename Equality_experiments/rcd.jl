function rcd(path, seed, X, y, f, g, fStar, Li, n, d, maxIter, trackIter)
    w0 = zeros(d, 1)
    # Random coordinate descent
    @printf("Random selection\n")
    if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter)* "randomL.jld")
        randomL = load(path * "_seed" * string(seed) *"iter" * string(maxIter)* "randomL.jld")
        if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter)* "randomLfstar.jld")
            randomLfstar = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "randomLfstar.jld")
        end
    else
        w = copy(w0)
        randomL = []
        randomLfstar = []
        push!(randomL, (f(w)))
        push!(randomLfstar, (f(w) - fStar))
        for i in 1:maxIter
            grad = g(w)
            j1 = rand(1:d)
            j2 = rand(1:d)
            dir = -(grad[j1] - grad[j2]) / (Li[j1] + Li[j2])
            w[j1] += dir
            w[j2] -= dir
            if mod(i, trackIter) == 0
                @printf("Iteration %d, Function = %f\n", i, (f(w)))
                push!(randomL, (f(w)))
                push!(randomLfstar, (f(w) - fStar))
            end
        end
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter)* "randomL.jld"; randomL)
        randomL = load(path * "_seed" * string(seed) * "iter" * string(maxIter)*  "randomL.jld")
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter)* "randomLfstar.jld"; randomLfstar)
        randomLfstar = load(path * "_seed" * string(seed) *"iter" * string(maxIter)* "randomLfstar.jld")
    end
    results_df = DataFrame(randomL)
    results_df = hcat(results_df, DataFrame(randomLfstar))
    return results_df
end

