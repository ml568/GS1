function GSq_project(w, g, L, lower, upper,y)
    d = length(w)
    lower = -upper
    maxDelta = -100
    maxj1 = 0
    maxj2 = 0
    for j1 in 1:d
        for j2 in 1:d
            delta = min((g[j1] - g[j2]) / (2 * L), w[j1] - lower, upper - w[j2])
            if delta > maxDelta
                maxDelta = delta
                maxj1 = j1
                maxj2 = j2
            end
        end
    end
    j1 = maxj1
    j2 = maxj2
    dir = -min((g[j1] - g[j2]) / (2 * L), w[j1] - lower, upper - w[j2])
    w[j1] += dir
    w[j2] -= dir
    alpha = w ./ y
    return alpha, w
end