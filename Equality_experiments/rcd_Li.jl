function rcd_Li(path, seed, results_df, X, y, f, g, fStar, Li, n, d, maxIter, trackIter)
    w0 = zeros(d, 1)
    # Random coordinate descent Li
    @printf("Random selection Li \n")
    if isfile(path * "_seed" * string(seed) *  "iter" * string(maxIter)*"randomLi.jld")
        randomLi = load(path * "_seed" * string(seed) *  "iter" * string(maxIter)*"randomLi.jld")
        if isfile(path * "_seed" * string(seed) *  "iter" * string(maxIter)*"randomLifstar.jld")
            randomLifstar = load(path * "_seed" * string(seed) *  "iter" * string(maxIter)*"randomLifstar.jld")
        end
    else
        w = copy(w0)
        randomLi = []
        randomLifstar = []
        push!(randomLi, (f(w)))
        push!(randomLifstar, (f(w) - fStar))

        p = Li ./ sum(Li)
        for i in 1:maxIter
            grad = g(w)
            j1 = findfirst(cumsum(p[:]) .> rand())
            j2 = findfirst(cumsum(p[:]) .> rand())
            dir = -(grad[j1] - grad[j2]) / (Li[j1] + Li[j2])
            w[j1] += dir
            w[j2] -= dir
            if mod(i, trackIter) == 0
                @printf("Iteration %d, Function = %f\n", i, (f(w)))
                push!(randomLi, (f(w)))
                push!(randomLifstar, (f(w) - fStar))
            end
        end
        jldsave(path * "_seed" * string(seed) *  "iter" * string(maxIter)*"randomLi.jld"; randomLi)
        randomLi = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "randomLi.jld")
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter)* "randomLifstar.jld"; randomLifstar)
        randomLifstar = load(path * "_seed" * string(seed) * "iter" * string(maxIter)* "randomLifstar.jld")
    end
    results_df = hcat(results_df, DataFrame(randomLi))
    results_df = hcat(results_df, DataFrame(randomLifstar))
    return results_df
end