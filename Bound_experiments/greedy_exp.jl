using Printf
using LinearAlgebra
using Random
using Plots
using DataFrames
using CSV
using JLD2
using StatsBase
using RDatasets
using Missings
using Statistics
# using MLJ

include("GS1.jl")
# include("GSS.jl")
# include("GSqbound.jl")
include("real_data.jl")
path = "greedyRandomdatajld/"
using Random
# include("utils.jl")
seeds = [25, 1, 2, 3, 4, 5]
for seed in seeds
    Random.seed!(seed)
    ds = "random"  #Choose from LifeExpectancy,random, Boston, RB, housing
    X, y = load_data(ds)
    n, d = size(X)
    z = size(y, 2)
    w = randn(d, z)

    # parameters 
    allGS = true
    if allGS
        maxIter = 20000
        marker_iter = 2000
    else
        maxIter = 50000
    end
    trackIter = 1
    upper = 1
    lower = -1
    weirdInit = false

    f(w) = (1 / 2) * norm(X * w - y)^2
    g(w) = X' * (X * w - y)
    # fStar = f(X \ y)

    # Compute L 
    E = eigen(X' * X)
    L = maximum(E.values)
    Li = diag(X' * X)
    L2 = 2 * maximum(Li)
    w0 = zeros(d, z)
    if weirdInit
        w0[1:Integer(d / 2)] .= lower + 1e-10
        w0[Integer(d / 2)+1:end] .= upper - 1e-10
    end

    gs1 = []
    gss = []
    ran = []
    gsq = []
    count_gs1 = []
    count_gss = []
    count_gsq = []
    count_ran = []
    total_gs1 = []

    # GS1
    w = copy(w0)
    # Estimating fstar

    suff_name = "n" * string(n) * "d" * string(d) * "seed" * string(seed) * "iter" * string(maxIter)
    # suff_name = ds * "seed" * string(seed)
    fname = path * "fStar_" * suff_name
    # if file exists, load the file
    if isfile(fname)
        fStar = load(fname)
        fStar = fStar["fStar"]
    else
        @printf "Estimating fstar"
        for i in 1:(20*maxIter)
            grad = g(w)
            grad = dropdims(grad, dims=2)
            what, _, _ = GS1_project(w, grad, L2 / 2, lower, upper)
            # what, _, _ = GS1_project(w, grad, Li, lower, upper)
            w[:] = what
        end
        fStar = f(w)
        jldsave(fname; fStar)
        fStar = load(fname)
        fStar = fStar["fStar"]
    end
    if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter) * "gs1.jld")
        gs1 = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "gs1.jld")
        gs1 = gs1["gs1"]
        total_gs1 = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "total_gs1.jld")
        count_gs1 = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "count_gs1.jld")
        count_gs1 = count_gs1["count_gs1"]
        total_gs1 = total_gs1["total_gs1"]
    else
        w = copy(w0)
        push!(gs1, (f(w) - fStar))
        # push!(gs1, f(w))
        @printf "Running GS1"
        for i in 1:maxIter
            grad = g(w)
            grad = dropdims(grad, dims=2)
            GS1_interior = 0
            gs1_tc = 0
            what, GS1_interior, gs1_tc = GS1_project(w, grad, L2 / 2, lower, upper)
            w[:] = what
            push!(count_gs1, GS1_interior)
            push!(total_gs1, gs1_tc)

            if mod(i, trackIter) == 0
                @printf("Iteration %d, Function = %f\n", i, (f(w) - fStar))
                push!(gs1, (f(w) - fStar))
                # push!(gs1, f(w))
            end
        end
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter) * "gs1.jld"; gs1)
        gs1 = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "gs1.jld")
        gs1 = gs1["gs1"]
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter) * "count_gs1.jld"; count_gs1)
        jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter) * "total_gs1.jld"; total_gs1)
        total_gs1 = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "total_gs1.jld")
        count_gs1 = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "count_gs1.jld")
        count_gs1 = count_gs1["count_gs1"]
        total_gs1 = total_gs1["total_gs1"]
    end
    # sleep(20)
    # GSS
    if allGS
        if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter) * "gss.jld")
            gss = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "gss.jld")
            count_gss = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "count_gss.jld")
            gss = gss["gss"]
            count_gss = count_gss["count_gss"]
        else
            w = copy(w0)
            push!(gss, (f(w) - fStar))
            @printf "Running GSS"
            for i in 1:maxIter
                GSS_interior = 0
                grad = g(w)
                gradPos = copy(grad)
                gradNeg = copy(grad)
                gradPos[w.<=lower] .= -Inf
                gradNeg[w.>=upper] .= Inf
                j1 = argmax(gradPos)
                j2 = argmin(gradNeg)
                dir = -min((grad[j1] - grad[j2]) / (2 * L2), w[j1] - lower, upper - w[j2])
                # dir = -min((grad[j1] - grad[j2]) / (Li[j1] + Li[j2]), w[j1] - lower, upper - w[j2])
                w[j1] += dir
                # GSS_interior += 1
                w[j2] -= dir
                # GSS_interior += 1

                # count the number of varibles that are inside the bounds
                # interior_vars = (w .> lower) .& (w .< upper)
                interior_vars_gss = (w .- lower .> 10^-6) .& (w .- upper .< -10^-6)
                GSS_interior = sum(interior_vars_gss)

                push!(count_gss, GSS_interior)

                if mod(i, trackIter) == 0
                    @printf("Iteration %d, Function = %f\n", i, (f(w) - fStar))
                    push!(gss, (f(w) - fStar))
                end
            end
            jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter) * "gss.jld"; gss)
            jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter) * "count_gss.jld"; count_gss)
            count_gss = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "count_gss.jld")
            gss = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "gss.jld")
            gss = gss["gss"]
            count_gss = count_gss["count_gss"]
        end
    end
    if allGS
        if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter) * "ran.jld")
            ran = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "ran.jld")
            count_ran = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "count_ran.jld")
            ran = ran["ran"]
            count_ran = count_ran["count_ran"]
        else
            w = copy(w0)
            push!(ran, (f(w) - fStar))
            for i in 1:maxIter
                ran_interior = 0
                grad = g(w)
                j1 = rand(1:d)
                j2 = rand(1:d)
                dir = -min((grad[j1] - grad[j2]) / (L2) / 4, w[j1] - lower, upper - w[j2])
                # dir = -min((grad[j1] - grad[j2]) / ((Li[j1] + Li[j2])) / 4, w[j1] - lower, upper - w[j2])
                w[j1] += dir
                w[j2] -= dir
                # count the number of varibles that are inside the bounds

                interior_vars_ran = (w .- lower .> 10^-6) .& (w .- upper .< -10^-6)
                ran_interior = sum(interior_vars_ran)

                push!(count_ran, ran_interior)
                if mod(i, trackIter) == 0
                    @printf("Iteration %d, Function = %f\n", i, (f(w) - fStar))
                    push!(ran, (f(w) - fStar))
                end
            end
            jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter) * "ran.jld"; ran)
            jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter) * "count_ran.jld"; count_ran)
            count_ran = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "count_ran.jld")
            ran = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "ran.jld")
            ran = ran["ran"]
            count_ran = count_ran["count_ran"]
        end
    end

    if allGS
        if isfile(path * "_seed" * string(seed) * "iter" * string(maxIter) * "gsq.jld")
            gsq = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "gsq.jld")
            count_gsq = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "count_gsq.jld")
            gsq = gsq["gsq"]
            count_gsq = count_gsq["count_gsq"]
        else
            w = copy(w0)
            push!(gsq, (f(w) - fStar))
            for i in 1:maxIter
                GSq_interior = 0
                grad = g(w)
                maxDelta = -100
                maxj1 = 0
                maxj2 = 0
                for j1 in 1:d
                    for j2 in 1:d
                        delta = min((grad[j1] - grad[j2]) / (2 * L2), w[j1] - lower, upper - w[j2])
                        # delta = min((grad[j1] - grad[j2])^2 / (Li[j1] + Li[j2]) , w[j1] - lower, upper - w[j2])
                        if delta > maxDelta
                            maxDelta = delta
                            maxj1 = j1
                            maxj2 = j2
                        end
                    end
                end
                j1 = maxj1
                j2 = maxj2
                dir = -min((grad[j1] - grad[j2]) / (2 * L2), w[j1] - lower, upper - w[j2])
                # dir = -min(((grad[j1] - grad[j2]) / (Li[j1] + Li[j2])), w[j1] - lower, upper - w[j2])
                w[j1] += dir
                # GSq_interior += 1
                w[j2] -= dir
                # GSq_interior += 1

                # count the number of varibles that are inside the bounds
                # interior_vars = (w .> lower) .& (w .< upper)
                interior_vars = (w .- lower .> 10^-6) .& (w .- upper .< -10^-6)
                GSq_interior = sum(interior_vars)

                push!(count_gsq, GSq_interior)
                if mod(i, trackIter) == 0
                    @printf("Iteration %d, Function = %f\n", i, (f(w) - fStar))
                    push!(gsq, (f(w) - fStar))
                end
            end
            jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter) * "gsq.jld"; gsq)
            jldsave(path * "_seed" * string(seed) * "iter" * string(maxIter) * "count_gsq.jld"; count_gsq)
            gsq = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "gsq.jld")
            count_gsq = load(path * "_seed" * string(seed) * "iter" * string(maxIter) * "count_gsq.jld")
            gsq = gsq["gsq"]
            count_gsq = count_gsq["count_gsq"]
        end
    end

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

    # Filter data
    f1 = point_of_zero(ran)
    f2 = point_of_zero(gs1)
    f3 = point_of_zero(gss)
    f4 = point_of_zero(gsq)
    plot_iter = minimum([f1, f2, f3, f4]) - 1

    # plot gs1
    if allGS
        p1 = plot(1:trackIter:plot_iter, ran[1:plot_iter], linewidth=5, linecolor=:royalblue3, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", xlabel="Iteration", ylabel="log(f(x)-f(*))", yaxis=:log10)
        scatter!(p1, 1:marker_iter:plot_iter, ran[(0:marker_iter:maxIter).+1], label="", marker=(:circle, 6, :lightblue), yaxis=:log10)
        plot!(p1, [NaN], [NaN], line=(:royalblue3, 5), marker=(:circle, 15, 0.9, :lightblue), label="Random")

        p2 = plot!(1:trackIter:plot_iter, gs1[1:plot_iter], linewidth=5, linecolor=:orangered3, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", yaxis=:log10)
        scatter!(p2, 1:marker_iter:plot_iter, gs1[(0:marker_iter:maxIter).+1], label="", marker=(:rect, 6, :orange), yaxis=:log10)
        plot!(p2, [NaN], [NaN], line=(:orangered3, 5), marker=(:rect, 15, 0.9, :orange), label="GS-1")

        p3 = plot!(1:trackIter:plot_iter, gss[1:plot_iter], linewidth=5, linecolor=:green, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", yaxis=:log10)
        scatter!(p3, 1:marker_iter:plot_iter, gss[(0:marker_iter:maxIter).+1], label="", marker=(:star5, 6, :greenyellow), yaxis=:log10)
        plot!(p3, [NaN], [NaN], line=(:green, 5), marker=(:star5, 15, 0.9, :greenyellow), label="GS-s")

        p4 = plot!(1:trackIter:plot_iter, gsq[1:plot_iter], linewidth=5, linecolor=:darkorchid3, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", yaxis=:log10)
        scatter!(p4, 1:marker_iter:plot_iter, gsq[(0:marker_iter:maxIter).+1], label="", marker=(:diamond, 6, :plum2), yaxis=:log10)
        plot!(p4, [NaN], [NaN], line=(:darkorchid3, 5), marker=(:diamond, 15, 0.9, :plum2), label="GS-q")
        # plot!(0:trackIter:maxIter, gss, label="GS-s", linewidth=5, thickness_scaling=1, yguidefontsize=16, xguidefontsize=16, xtickfontsize=16, ytickfontsize=16, legendfontsize=16)
        # plot!(0:trackIter:maxIter, gsq, label="GS-q", linewidth=5, thic\]kness_scaling=1, yguidefontsize=16, xguidefontsize=16, xtickfontsize=16, ytickfontsize=16, legendfontsize=16)
        # savefig("gssvsgs1gsq_wifalse_seed1_RB.pdf")
        savefig("results/" * "gssvsgs1gsq_wifalse_seed_" * string(seed) * "iter" * string(maxIter) * ds * ".pdf")
    end
    if allGS
        p5 = plot(1:maxIter, count_ran, linewidth=5, linecolor=:royalblue3, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", xlabel="Iteration", ylabel="Number of interior points", yaxis=:log10)
        scatter!(p5, 1:marker_iter:maxIter, count_ran[(1:marker_iter:maxIter)], label="", marker=(:circle, 6, :lightblue), yaxis=:log10)
        plot!(p5, [NaN], [NaN], line=(:royalblue3, 5), marker=(:circle, 15, 0.9, :lightblue), label="Random interior variables")

        p6 = plot!(1:maxIter, count_gs1, linewidth=5, linecolor=:orangered3, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", yaxis=:log10)
        scatter!(p6, 1:marker_iter:maxIter, count_gs1[(1:marker_iter:maxIter)], label="", marker=(:rect, 6, :orange), yaxis=:log10)
        plot!(p6, [NaN], [NaN], line=(:orangered3, 5), marker=(:rect, 15, 0.9, :orange), label="GS-1 interior variables")

        p7 = plot!(1:maxIter, count_gss, linewidth=5, linecolor=:green, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", yaxis=:log10)
        scatter!(p7, 1:marker_iter:maxIter, count_gss[(1:marker_iter:maxIter)], label="", marker=(:star5, 6, :greenyellow), yaxis=:log10)
        plot!(p7, [NaN], [NaN], line=(:green, 5), marker=(:star5, 15, 0.9, :greenyellow), label="GS-s interior variables")

        p8 = plot!(1:maxIter, count_gsq, linewidth=5, linecolor=:darkorchid3, thickness_scaling=1, xtickfontsize=9, ytickfontsize=9,
            legendfontsize=9, label="", yaxis=:log10)
        scatter!(p8, 1:marker_iter:maxIter, count_gsq[(1:marker_iter:maxIter)], label="", marker=(:diamond, 6, :plum2), yaxis=:log10)
        plot!(p8, [NaN], [NaN], line=(:darkorchid3, 5), marker=(:diamond, 15, 0.9, :plum2), label="GS-q interior variables")
        savefig("results/" * "updated_points_wifalse_seed_" * string(seed) * "iter" * string(maxIter) * ds * ".pdf")
    end
    # plot interior points
    # p5= plot(1:maxIter, count_gs1, linewidth=5, thickness_scaling=1, yguidefontsize=16, xguidefontsize=16, xtickfontsize=16, ytickfontsize=16, legendfontsize=16,
    #     label="GS-1 interior variables", xlabel="Iteration", ylabel="Number of interior points")
    # plot!(1:maxIter, count_gss, linewidth=5, thickness_scaling=1, yguidefontsize=16, xguidefontsize=16, xtickfontsize=16, ytickfontsize=16, legendfontsize=16, label="GS-s interior variables")
    # plot!(1:maxIter, count_gsq, linewidth=5, thickness_scaling=1, yguidefontsize=16, xguidefontsize=16, xtickfontsize=16, ytickfontsize=16, legendfontsize=16, label="GS-q interior variables")

    # # p6 = plot(p2, p3, p4, p5, layout=(2, 2))
    # # p6 = plot(p1, p5, layout=(1, 2), legend=false)
    # # savefig("updated_points_wifalse_seed1_RB.pdf")
    # savefig("results/" * "updated_points_wifalse_seed_" * string(seed) * "iter" * string(maxIter) * ds * ".pdf")


    plot(1:maxIter, total_gs1, linewidth=5, thickness_scaling=1, xtickfontsize=16, ytickfontsize=16, yguidefontsize=16, xguidefontsize=16,
        legendfontsize=16, ylabel="No. of variables updated", xlabel="Iteration", legend=false)
    # savefig("gs1_no_variables_updated_false_seed1_RB.pdf")
    savefig("results/" * "gs1_no_variables_updated_false_seed_" * string(seed) * "iter" * string(maxIter) * ds * ".pdf")
    @printf("The percentage of 2 coordinate updates by GS-1 is %f\n", (countmap(total_gs1)[2] / maxIter) * 100)
    # histogram(total_gs1,
    #     histtype = "bar",
    #     xlabel="No. of variables updated",
    #     ylabel="Frequency",
    #     linewidth=3,
    #     thickness_scaling=1,
    #     xtickfontsize=16,
    #     ytickfontsize=16,
    #     yguidefontsize=16,
    #     xguidefontsize=16,
    #     legendfontsize=16,
    #     legend=false,
    #     bins=4,  # Adjust the number of bins as needed
    # )
    # Create a bar plot with bars centered on unique values
    # Calculate the unique values and bin counts
    unique_values = unique(total_gs1)
    bin_counts = [count(x -> x == val, total_gs1) for val in unique_values]
    bar(
        unique_values,
        bin_counts,
        xlabel="No. of variables updated",
        ylabel="Frequency",
        linewidth=3,
        thickness_scaling=1,
        xtickfontsize=16,
        ytickfontsize=16,
        yguidefontsize=16,
        xguidefontsize=16,
        legendfontsize=16,
        legend=false,
    )

    # Save the plot as a PDF file
    savefig("results/" * "hist_gs1" * string(seed) * "iter" * string(maxIter) * ds * ".pdf")
end