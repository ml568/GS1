using LinearAlgebra
include("./svm_obj.jl")
# using .svm_obj
# using svm_dual_objective_kernel, svm_dual_gradient_kernel, rbf_kernel

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

function svm_dual_gradient(X, y, alpha)
    n = size(X, 1)
    m = size(X, 2)

    P = Matrix{Float64}(undef, n, n)
    for i in 1:n
        for j in 1:n
            P[i, j] = y[i] * y[j] * dot(X[i, :], X[j, :])
        end
    end

    gradient = ones(n) - P * alpha

    return gradient
end

function numerical_gradient(f, x; ε=1e-6)
    n = length(x)
    grad = zeros(n)
    for i in 1:n
        x[i] += ε
        f_plus = f(x)

        x[i] -= 2ε
        f_minus = f(x)

        grad[i] = (f_plus - f_minus) / (2ε)

        x[i] += ε  # Reset the value of x[i]
    end
    return grad
end

function gradient_checker(f, grad_f, x; ε=1e-6, tol=1e-5)
    analytic_grad = grad_f(x)
    numeric_grad = numerical_gradient(f, x, ε=ε)

    diff = norm(analytic_grad - numeric_grad) / (norm(analytic_grad) + norm(numeric_grad))

    if diff < tol
        println("Gradient is correct!")
    else
        println("Gradient seems incorrect!")
        println("Difference: ", diff)
    end
end


# Test
n, m = 10, 5  # Example sizes
X_test = randn(n, m)
y_test = rand([-1, 1], n)
alpha_test = rand(n)

# Wrap the functions to pass X and y
# objective_wrapper(alpha) = svm_dual_objective(X_test, y_test, alpha)
# gradient_wrapper(alpha) = svm_dual_gradient(X_test, y_test, alpha)


# Testing the objectives and gradients with kernel
K = rbf_kernel(X_test, X_test, 0.5)
objective_wrapper(alpha) = svm_dual_objective_kernel(K, y_test, alpha)
gradient_wrapper(alpha) = svm_dual_gradient_kernel(K, y_test, alpha)
gradient_checker(objective_wrapper, gradient_wrapper, alpha_test)
