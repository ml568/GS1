function GS1_project(x, g, L, lower, upper)
    # sorting the gradient
    sortidx = sortperm(g, rev=true)
    g .= g[sortidx]

    x_orig = x[:]
    x .= x[sortidx]
    n = length(x)
    orig = collect(1:n)
    d = zeros(n, 1)


    sum_xlower = 0
    sum_xupper = 0

    i = 1
    j = n
    iter = 1    
    objs = []
    count_interior = 0
    counter = 0

    while true
        delta = (g[i] - g[j]) / (4 * L)
        # delta = (g[i] - g[j])  / (sqrt(Li[i]) + sqrt(Li[j]))

        if (-sum_xlower + delta < 0) && (-sum_xupper + delta < 0)
            # println("Neither variable $i or $j want to move, ")
            if sum_xlower < sum_xupper
                # println("moving lower variable $i")
                d[i] = sum_xlower - sum_xupper
                count_interior += 1
                counter += 1
                break
            else
                # println("moving upper variable $j")
                d[j] = sum_xlower - sum_xupper
                count_interior += 1
                counter += 1
                break
            end
        elseif -sum_xlower + delta < 0
            # println("Lower variable $i does not want to move")
            d[j] = sum_xlower - sum_xupper # done
            count_interior += 1
            counter += 1
            break
        elseif -sum_xupper + delta < 0
            # println("Upper variable $j does not want to move")
            d[i] = sum_xlower - sum_xupper # done
            count_interior += 1
            counter += 1
            break
        end

        if (x[i] + sum_xlower - delta >= lower) && (x[j] - sum_xupper + delta <= upper)
            # println("Variables $i and $j moving to interior")
            d[i] = sum_xlower - delta # done
            count_interior += 1
            counter += 1
            d[j] = -sum_xupper + delta # done
            count_interior += 1
            counter += 1
            break
        end # done

        if (x[i] + sum_xlower - delta < lower) && (x[j] - sum_xupper + delta > upper)
            # println("Both $i and $j want to leave set, ")
            diff1 = lower - (x[i] + sum_xlower - delta)
            diff2 = (x[j] - sum_xupper + delta) - upper
            if diff1 > diff2
                # println("moving lower variable $i to bound")
                d[i] = lower - x[i] # done
                counter += 1
                sum_xlower += x[i] - lower
                i += 1
            else
                # println("moving upper variable $j to bound")
                d[j] = upper - x[j] # done
                counter += 1
                sum_xupper += upper - x[j]
                j -= 1
            end
        elseif x[i] + sum_xlower - delta < lower
            # println("Moving lower variable $i to bound")
            d[i] = lower - x[i] # done
            counter += 1
            sum_xlower += x[i] - lower # done
            i += 1
        else
            # println("Moving upper variable $j to bound")
            d[j] = upper - x[j] # done
            counter += 1
            sum_xupper += upper - x[j] # done
            j -= 1
        end
    end

    if any(x .+ d .> upper + 1e-4) || any(x .+ d .< lower - 1e-4)
        # println(x + d)
        println("Weird case")
        sleep(0.5)
    end
    x = x + d

    num_vars_updated = sum(abs.(d) .> 10^-6)
    interior_vars = (x .- lower .> 10^-6) .& (x .- upper .< -10^-6)
    num_interior_vars = sum(interior_vars)

    # now recover the original values
    x_orig[sortidx] .= x
    return x_orig, num_interior_vars, num_vars_updated
end
