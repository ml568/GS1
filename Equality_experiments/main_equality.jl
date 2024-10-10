# Requires "dataset" and "seeds" and "maxIter" to be specified

using Printf
using LinearAlgebra
using Random
using Plots
using DataFrames
using JLD2
using CodecZlib
using CSV
using StatsBase
using RDatasets
using Missings
using Statistics
include("rcd.jl")
include("gcd.jl")
include("rcd_Li.jl")
include("gsq.jl")
include("gs1.jl")
include("gcdratio.jl")

# seeds = [1,2,3,4,5,6,7,8,9,10]
# seeds = [30, 50, 1, 2, 3]
#seeds = [30]
ds = ["randomdata"]
for seed in seeds
    Random.seed!(seed)
    # parameters 
    marker_iter = Integer(maxIter/10)
    trackIter = 1

    # Make function
    if dataset == "syn1"
	n = 1000
	d = 1000
	    X = randn(n, d)
    w = rand(d, 1)
    y = X * w + randn(n, 1)
	elseif dataset == "syn2"
		n = 1000
		d = 1000
    		X = randn(n, d) * diagm(rand(d))
    w = rand(d, 1)
    y = X * w + randn(n, 1)
	else
			(X,y) = load("../datasets/"*dataset*".jld","X","y")
			(n,d) = size(X)
			y[y.==0] .= -1
	y[y.==2] .= -1
	y[y.>2] .= 1
	end
    path = "plots/"*dataset

    # data = CSV.File("data/Residential-Building-Data-Set.csv") |> DataFrame
    # # Convert specific columns to Float64
    # Xtemp = parse.(Float64, data[2:end, 5:end-2])
    # X = Matrix(Xtemp)
    # # X = Matrix(data[3:end, 7:end-2])
    # y = parse.(Float64, data[2:end, end-1])
    # dt = fit(ZScoreTransform, X, dims=2)
    # X = StatsBase.transform(dt, X)
    # n, d = size(X)
    # z = size(y, 2)
    # w = randn(d, z)

    f(w) = (1 / 2) * norm(X * w - y)^2
    g(w) = X' * (X * w - y)

    # fStar = f(X \ y)

    # Compute L 
    E = eigen(X' * X)
    L = maximum(E.values)
    Li = diag(X' * X)
    w0 = zeros(d, 1)

    # Experiments
    results_df = DataFrame()

    # Estimating fstar
    @printf("Estimating fstar\n")
    suff_name = "n" * string(n) * "d" * string(d) * "seed" * string(seed) * "iter" * string(maxIter)
    # fname = path * "fStar_randgreed" * suff_name
    fname = path * "fStar_randgreed" * suff_name
    fname = path * string(seed) * string(maxIter) * "fStar.jld"
    # if file exists, load the file
    if isfile(fname)
        fStar = load(fname)
        fStar = fStar["fStar"]
    else
        w = copy(w0)
        for i in 1:20*maxIter
            grad = g(w)
            mu = sum(grad) / length(grad)
            j1 = argmax((grad .- mu) ./ sqrt.(Li))
            j2 = argmin((grad .- mu) ./ sqrt.(Li))
            dir = -(grad[j1] - grad[j2]) / (Li[j1] + Li[j2])
            w[j1] += dir
            w[j2] -= dir
            @printf("Iteration %d, Function = %f\n", i, (f(w)))
        end
        fStar = f(w)
        jldsave(fname; fStar)
        fStar = load(fname)
        fStar = fStar["fStar"]
    end

    results_df = rcd(path, seed, X, y, f, g, fStar, Li, n, d, maxIter, trackIter) #random coordinate descent
    @show size(X)
    @show size(y)
    results_df = gcd(path, seed, results_df, X, y, f, g, fStar, Li, n, d, maxIter, trackIter) #greedy coordinate descent
    results_df = rcd_Li(path, seed, results_df, X, y, f, g, fStar, Li, n, d, maxIter, trackIter) #random Li coordinate descent
    results_df = gsq(path, seed, results_df, X, y, f, g, fStar, Li, n, d, maxIter, trackIter) #gsq
    results_df = gs1(path, seed, results_df, X, y, f, g, fStar, Li, n, d, maxIter, trackIter) #gs1
    results_df = gcdratio(path, seed, results_df, X, y, f, g, fStar, Li, n, d, maxIter, trackIter) #ratio

    function point_of_zero(f_fstar)
        zero_indices = findall(x -> x <= 0, f_fstar)
        if isempty(zero_indices)
            # Return a large number (e.g., Inf) if the result is empty
            return maxIter
        else
            # Calculate the minimum value
            return minimum(zero_indices)
        end
    end


    function plot_results(results_df)

        # Filter data
        f1 = point_of_zero(results_df.randomLfstar)
        f2 = point_of_zero(results_df.greedyLfstar)
        f3 = point_of_zero(results_df.randomLifstar)
        f4 = point_of_zero(results_df.greedyLifstar)
        f5 = point_of_zero(results_df.greedyLigs1fstar)
        f6 = point_of_zero(results_df.greedyLiApxfstar)
        plot_iter = minimum([f1, f2, f3, f4, f5, f6]) - 1

        # Plot
        p1 = plot(1:trackIter:plot_iter, results_df.randomLfstar[1:plot_iter], linewidth=5, linecolor=:royalblue3, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", xlabel="Iteration", ylabel="log(f(x)-f(*))", yaxis=:log10)
        scatter!(p1, 1:marker_iter:plot_iter, results_df.randomLfstar[(0:marker_iter:maxIter).+1], label="", marker=(:circle, 6, :lightblue), yaxis=:log10)
        plot!(p1, [NaN], [NaN], line=(:royalblue3, 5), marker=(:circle, 15, 0.9, :lightblue), label="Random")

        p2 = plot!(1:trackIter:plot_iter, results_df.greedyLfstar[1:plot_iter], linewidth=5, linecolor=:orangered3, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", yaxis=:log10)
        scatter!(p2, 1:marker_iter:plot_iter, results_df.greedyLfstar[(0:marker_iter:maxIter).+1], label="", marker=(:rect, 6, :orange), yaxis=:log10)
        plot!(p2, [NaN], [NaN], line=(:orangered3, 5), marker=(:rect, 15, 0.9, :orange), label="Greedy")

        p3 = plot!(1:trackIter:plot_iter, results_df.randomLifstar[1:plot_iter], linewidth=5, linecolor=:green, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", yaxis=:log10)
        scatter!(p3, 1:marker_iter:plot_iter, results_df.randomLifstar[(0:marker_iter:maxIter).+1], label="", marker=(:star5, 6, :greenyellow), yaxis=:log10)
        plot!(p3, [NaN], [NaN], line=(:green, 5), marker=(:star5, 15, 0.9, :greenyellow), label="Random Li")

        p4 = plot!(1:trackIter:plot_iter, results_df.greedyLifstar[1:plot_iter], linewidth=5, linecolor=:darkorchid3, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", yaxis=:log10)
        scatter!(p4, 1:marker_iter:plot_iter, results_df.greedyLifstar[(0:marker_iter:maxIter).+1], label="", marker=(:diamond, 6, :plum2), yaxis=:log10)
        plot!(p4, [NaN], [NaN], line=(:darkorchid3, 5), marker=(:diamond, 15, 0.9, :plum2), label="Greedy Li (GS-q)")


        p5 = plot!(1:trackIter:plot_iter, results_df.greedyLigs1fstar[1:plot_iter], linewidth=5, linecolor=:goldenrod3, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", yaxis=:log10)
        scatter!(p5, 0:marker_iter:plot_iter, results_df.greedyLigs1fstar[(0:marker_iter:maxIter).+1], label="", marker=(:hexagon, 6, :gold), yaxis=:log10)
        plot!(p5, [NaN], [NaN], line=(:goldenrod3, 5), marker=(:hexagon, 15, 0.9, :gold), label="Greedy Li (GS-1)")

        p6 = plot!(1:trackIter:plot_iter, results_df.greedyLiApxfstar[1:plot_iter], linewidth=5, linecolor=:cyan4, thickness_scaling=1, xtickfontsize=12, ytickfontsize=12,
            legendfontsize=10, label="", yaxis=:log10)
        scatter!(0:marker_iter:plot_iter, results_df.greedyLiApxfstar[(0:marker_iter:maxIter).+1], label="", marker=(:utriangle, 6, :turquoise), yaxis=:log10)
        plot!(p6, [NaN], [NaN], line=(:cyan4, 5), marker=(:utriangle, 15, 0.9, :turquoise), label="Greedy Li (Ratio)")

        #savefig("plots/" * "fStar_randgreed" * suff_name * "iter" * string(maxIter) * ".pdf")
        savefig(path * string(seed) * string(maxIter) * ".pdf")
        # Plot
        # p7 = plot(0:trackIter:maxIter, results_df.randomL; rcd_plot..., plot_opts...)
        # p8 = plot!(0:trackIter:maxIter, results_df.greedyL; gcd_plot..., plot_opts...)
        # p9 = plot!(0:trackIter:maxIter, results_df.randomLi; randomLi_plot..., plot_opts...)
        # p10 = plot!(0:trackIter:maxIter, results_df.greedyLi; gsq_plot..., plot_opts...)
        # p11 = plot!(0:trackIter:maxIter, results_df.greedyLigs1; gs1_plot..., plot_opts...)
        # p12 = plot!(0:trackIter:maxIter, results_df.greedyLiApx; gcdratio_plot..., plot_opts...)
        # savefig("plots/" * "randgreed" * suff_name * "iter" * string(maxIter) * ".pdf")
        # return [p1, p2, p3, p4, p5, p6]
    end
    plot_results(results_df)
end









# # Configs
# rcd_plot = Dict(pairs((
#     line=(:royalblue3, 5),
#     label=""
# )))

# gcd_plot = Dict(pairs((
#     line=(:orangered3, 5),
#     label="Greedy"
# )))

# randomLi_plot = Dict(pairs((
#     line=(:green, 5),
#     label="Random Li"
# )))

# gsq_plot = Dict(pairs((
#     line=(:darkorchid3, 5),
#     label="Greedy Li (GS-q)"
# )))

# gs1_plot = Dict(pairs((
#     line=(:goldenrod3, 5),
#     label="Greedy Li (GS-1)"
# )))

# gcdratio_plot = Dict(pairs((
#     line=(:cyan4, 5),
#     label="Greedy Li (Ratio)"
# )))

# plot_opts = Dict(
#     :xlabel => "Iterations",
#     :ylabel => "log(f(x)-f(*))",
#     :linewidth => 5,
#     :legend => :bottomright,
#     :legendfontsize => 9,
#     :thickness_scaling => 1,
#     :xtickfontsize => 16,
#     :ytickfontsize => 12,
#     :yaxis => :log10
# )