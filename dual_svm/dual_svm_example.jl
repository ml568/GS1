using LinearAlgebra
using Convex, SCS, COSMO, Mosek, MosekTools
using Random

function dual_objective(alpha, X, y, λ)
    n, d = size(X)
    dual_obj = sum(alpha[i] for i in 1:n) - 0.5 * sum(alpha[i] * alpha[j] * y[i] * y[j] * (X[i, :] ⋅ X[j, :]) for i in 1:n, j in 1:n) - λ * sum(alpha)
    return dual_obj
end

function svm_dual_objective(X, y, alpha)
    n = size(X, 1)
    m = size(X, 2)

    P = Matrix{Float64}(undef, n, n)
    for i in 1:n
        for j in 1:n
            P[i, j] = y[i] * y[j] * dot(X[i, :], X[j, :])
        end
    end
    objective = sum(alpha) - 0.5 * dot(alpha, P * alpha)

    return objective
end

function dual_loss(α, X, y)
    t = α .* y
    return sum(α) - t' * X * X' * t
end

function dual_loss_DCP(α, X, y)
    n, m = size(X)
    K = X * X' # Precompute X * X'

    t = α .* y
    # t = Variable(n)
    objective = sum(α[i] for i in 1:n) - quadform(t, K)

    # objective = maximize(sum(α) - quadform(α .* y, X * X'))
    return objective
end

function svm_dual_gradient(X, y, C, alpha)
    n = size(X, 1)
    m = size(X, 2)

    P = Matrix{Float64}(undef, n, n)
    for i in 1:n
        for j in 1:n
            P[i, j] = y[i] * y[j] * dot(X[i, :], X[j, :])
        end
    end

    gradient = -ones(n) + P * alpha

    return gradient
end

function predict(α, X, y, X_test)
    n, m = size(X)
    K = X * X' # Precompute X * X'
    y_pred = zeros(size(X_test, 1))

    for i in 1:size(X_test, 1)
        x_test = X_test[i, :]
        s = 0
        for j in 1:n
            s += α[j] * y[j] * dot(X[j, :], x_test)
        end
        y_pred[i] = sign(s)
    end

    return y_pred
end

# Testing


# # Define your dataset and labels
# X = [1 2; 2 3; 3 4]
# y = [1, -1, 1]

# # Set the regularization parameter
# C = 1.0

# # Set an initial guess for alpha
# n = length(y)
# alpha_guess = ones(length(y))

# # Compute the objective function
# objective = svm_dual_objective(X, y, C, alpha_guess)
# println("Objective: $objective")

# # Compute the gradient
# gradient = svm_dual_gradient(X, y, C, alpha_guess)
# println("Gradient: $gradient")

# # Solving using SCP

# alpha = Variable(n)
# # problem = minimize(svm_dual_objective(X, y, C, alpha))
# problem = maximize(dual_loss_DCP(alpha, X, y))
# problem.constraints += alpha >= 0
# problem.constraints += alpha <= C
# problem.constraints += sum(dot(alpha, y)) == 0
# # problem.constraints += y >= l
# # problem.constraints += y <= u
# # problem.constraints += ones(n, 1)' * (y - x) == 0

# solve!(problem, SCS.Optimizer, verbose=false, silent_solver=true)