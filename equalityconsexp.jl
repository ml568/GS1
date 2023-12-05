using Printf
using LinearAlgebra
using Random
using Plots
using DataFrames

Random.seed!(100)
# parameters 
n = 1000
d = 1000
maxIter = 100
trackIter = 5

# Make function
#X = [ones(n,1) randn(n,d-1)*diagm(rand(d-1))]
w = rand(d, 1)
# X = randn(n, d) * diagm(rand(d))
X = randn(n, d)
y = X * w + randn(n, 1)

f(w) = (1 / 2) * norm(X * w - y)^2
g(w) = X' * (X * w - y)

fStar = f(X \ y)

# Compute L 
E = eigen(X' * X);
L = maximum(E.values);
Li = diag(X' * X);
w0 = zeros(d, 1);

# Experiments
results_df = DataFrame();

# Random coordinate descent
@printf("Random selection\n")
w = copy(w0);
randomL = [];
push!(randomL, f(w) - fStar)
for i in 1:maxIter
    grad = g(w)
    j1 = rand(1:d)
    j2 = rand(1:d)
    dir = -(grad[j1] - grad[j2]) / (Li[j1] + Li[j2])
    w[j1] += dir
    w[j2] -= dir
    if mod(i, trackIter) == 0
        @printf("Iteration %d, Function = %f\n", i, f(w))
        push!(randomL, f(w) - fStar)
    end
end
results_df[!, "randomL"] = randomL;

# Greedy coordinate descent
@printf("Greedy selection\n")
w = copy(w0);
greedyL = [];
push!(greedyL, f(w) - fStar)
for i in 1:maxIter
    grad = g(w)
    j1 = argmax(grad)
    j2 = argmin(grad)
    dir = -(grad[j1] - grad[j2]) / (Li[j1] + Li[j2])
    w[j1] += dir
    w[j2] -= dir
    if mod(i, trackIter) == 0
        @printf("Iteration %d, Function = %f\n", i, f(w))
        push!(greedyL, f(w) - fStar)
    end
end
results_df[!, "greedyL"] = greedyL;
println(first(results_df, 5))


# Random coordinate descent Li
@printf("Random selection Li \n")
w = copy(w0);
randomLi = [];
push!(randomLi, f(w) - fStar)
p = Li ./ sum(Li);
for i in 1:maxIter
    grad = g(w)
    j1 = findfirst(cumsum(p[:]) .> rand())
    j2 = findfirst(cumsum(p[:]) .> rand())
    dir = -(grad[j1] - grad[j2]) / (Li[j1] + Li[j2])
    w[j1] += dir
    w[j2] -= dir
    if mod(i, trackIter) == 0
        @printf("Iteration %d, Function = %f\n", i, f(w))
        push!(randomLi, f(w) - fStar)
    end
end
results_df[!, "randomLi"] = randomLi;

# Greedy coordinate descent Li (exact)
@printf("Greedy selection Li (exact)\n")
w = copy(w0);
greedyLi = [];
push!(greedyLi, f(w) - fStar)
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
        @printf("Iteration %d, Function = %f\n", i, f(w))
        push!(greedyLi, f(w) - fStar)
    end
end
results_df[!, "greedyLi"] = greedyLi;
println(first(results_df, 5))

# Greedy coordinate descent Li (approximate)
@printf("Greedy selection Li (approximate)\n")
w = copy(w0);
greedyLiApx = [];
push!(greedyLiApx, f(w) - fStar)
for i in 1:maxIter
    grad = g(w)
    j1 = argmax(grad ./ sqrt.(Li))
    j2 = argmin(grad ./ sqrt.(Li))
    dir = -(grad[j1] - grad[j2]) / (Li[j1] + Li[j2])
    w[j1] += dir
    w[j2] -= dir
    if mod(i, trackIter) == 0
        @printf("Iteration %d, Function = %f\n", i, f(w))
        push!(greedyLiApx, f(w) - fStar)
    end
end
results_df[!, "greedyLiApx"] = greedyLiApx;
println(first(results_df, 5))

# Greedy coordinate descent Li (approximate 2)
@printf("Greedy selection Li (approximate 2)\n")
w = copy(w0);
greedyLiApx2 = [];
push!(greedyLiApx2, f(w) - fStar)
for i in 1:maxIter
    grad = g(w)
    if mod(i, 2) == 0
        j1 = argmax(grad ./ sqrt.(Li))
        j2 = argmax(((grad .- grad[j1]) .^ 2) ./ (Li .+ Li[j1]))
    else
        j1 = argmin(grad ./ sqrt.(Li))
        j2 = argmax(((grad .- grad[j1]) .^ 2) ./ (Li .+ Li[j1]))
    end
    dir = -(grad[j1] - grad[j2]) / (Li[j1] + Li[j2])
    w[j1] += dir
    w[j2] -= dir
    if mod(i, trackIter) == 0
        @printf("Iteration %d, Function = %f\n", i, f(w))
        push!(greedyLiApx2, f(w) - fStar)
    end
end
results_df[!, "greedyLiApx2"] = greedyLiApx2;
println(first(results_df, 5))

# Plot randoml and greedyL
plot(0:trackIter:maxIter, results_df.randomL, linewidth=5, thickness_scaling=1, xtickfontsize=16, ytickfontsize=12,
    legendfontsize=10, label="Random", xlabel="Iteration", ylabel="f(x)-f(x*)", yaxis=:log10, legend=:bottomleft)
plot!(0:trackIter:maxIter, results_df.greedyL, linewidth=5, thickness_scaling=1, xtickfontsize=12, ytickfontsize=12,
    legendfontsize=10, label="Greedy", yaxis=:log10)
plot!(0:trackIter:maxIter, results_df.randomLi, linewidth=5, thickness_scaling=1, xtickfontsize=12, ytickfontsize=12,
    legendfontsize=10, label="Random Li", yaxis=:log10)
plot!(0:trackIter:maxIter, results_df.greedyLi, linewidth=5, thickness_scaling=1, xtickfontsize=12, ytickfontsize=12,
    legendfontsize=10, label="Greedy Li (Exact)", yaxis=:log10)
plot!(0:trackIter:maxIter, results_df.greedyLiApx, linewidth=5, thickness_scaling=1, xtickfontsize=12, ytickfontsize=12,
    legendfontsize=10, label="Greedy Li (Ratio)", yaxis=:log10)
plot!(0:trackIter:maxIter, results_df.greedyLiApx2, linewidth=5, thickness_scaling=1, xtickfontsize=12, ytickfontsize=12,
    legendfontsize=10, label="Greedy Li (Switch)", yaxis=:log10)


# Update title and savefigure
# title!("Random vs Greedy coordinate selection")
savefig("randomvsgreedy2.pdf")