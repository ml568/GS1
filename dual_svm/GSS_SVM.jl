function GSS_project(x, g, L, lower, upper, y)
    """
    Implements the GSS projection algorithm.
    """
    lower = -upper
    gradPos = copy(g)
    gradNeg = copy(g)   
    gradPos[x[:,1].<= lower] .= -Inf
    gradNeg[x[:,1].>=upper] .= Inf
    j1 = argmax(gradPos)
    j2 = argmin(gradNeg)
    dir = -min((g[j1] - g[j2]) / (2 * L), x[j1] - lower, upper - x[j2])
    x[j1] += dir
    x[j2] -= dir
    # now recover the original values
    alpha = x ./ y
    return alpha, x

end