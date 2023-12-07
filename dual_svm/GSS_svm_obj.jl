using LinearAlgebra
using Random
using Printf
using CSV
using DataFrames
using Plots
using StatsBase
using Missings
using LIBSVM
using Optim
using RDatasets
include("preprocess_dataset.jl")
include("custom_solvers.jl")
include("GS1_svm.jl")
include("GSS_SVM.jl")
include("GSQ_SVM.jl")
# using solve_alpha_with_sum_constraint

seed = 10
Random.seed!(seed);
dataset_names = ["iris"]

gs1 = [];
gss = [];
gsq = [];


# # heart disease data
# # Load the dataset
# data = CSV.File("sonar.csv") |> DataFrame
# # Remove missing or NaN rows
# data = dropmissing(data, disallowmissing=true)
# # Extracting features (X) and target variable (y)
# X = Matrix(data[:, 1:end-1])
# y = data[:, end]

# #For Sonar dataset
# y_binarized = map(yi -> yi == "R" ? -1 : 1, y)
# dt = fit(ZScoreTransform, X, dims=2)
# X = StatsBase.transform(dt, X)
# Y = Diagonal(y_binarized)

# # Split ratio (e.g., 0.8 for 80% training and 20% testing)
# split_ratio = 0.8
# # Get the number of samples
# n_samples = size(X, 1)
# # Generate random indices for training and testing samples
# train_indices = randperm(n_samples)[1:floor(Int, split_ratio * n_samples)]
# test_indices = setdiff(1:n_samples, train_indices)
# # Split the data into training and testing sets
# X_train = X[train_indices, :]
# # y_train = Y[train_indices, train_indices]
# y_train = y_binarized[train_indices]
# X_test = X[test_indices, :]
# # y_test = Y[test_indices, test_indices]
# y_test = y_binarized[test_indices]




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

    gradient = ones(n) - P * alpha  # Fixed this to get the gradient correct

    return gradient[:, 1]
end

# function svm_dual_objective_kernel(K, y, alpha)
#     # Use this for implementing SVM with an explicit kernel
#     # Remember to use the appropriate gradient
#     n = size(K, 1)
#     P = Matrix{Float64}(undef, n, n)
#     for i in 1:n
#         for j in 1:n
#             P[i, j] = y[i] * y[j] * K[i, j]
#         end
#     end
#     objective = sum(alpha) - 0.5 * dot(alpha, P * alpha)
#     return objective
# end

# function svm_dual_gradient_kernel(K, y, alpha)
#     # To be used along with the dual kernel objective
#     n = size(K, 1)

#     P = Matrix{Float64}(undef, n, n)
#     for i in 1:n
#         for j in 1:n
#             P[i, j] = y[i] * y[j] * K[i, j]
#         end
#     end

#     gradient = ones(n) - P * alpha  # Fixed this to get the gradient correct

#     return gradient[:, 1]
# end

function svm_dual_objective_kernel(K, y, alpha; C=1)
    n = length(y)

    # Convert alpha and y to 2D arrays
    alpha_2d = reshape(alpha, n, 1)
    y_2d = reshape(y, n, 1)

    # Calculate the dual objective
    dual_obj = sum(alpha_2d) - 0.5 * dot(alpha_2d, alpha_2d .* (y_2d .* (y_2d' * K)'))
    regularization_term = -0.5 * dot(alpha_2d, alpha_2d)

    return dual_obj + C * regularization_term
end

function svm_dual_gradient_kernel(K, y, alpha; C=1)
    n = length(y)

    # Convert alpha and y to 2D arrays
    alpha_2d = reshape(alpha, n, 1)
    y_2d = reshape(y, n, 1)

    # Calculate the gradient of the dual objective
    gradient = ones(n) .- alpha_2d .* (y_2d .* (y_2d' * K)')

    # Subtract the alpha term and apply regularization parameter C
    gradient .-= alpha_2d
    gradient *= C

    # Reshape gradient back to 1-D array
    gradient_1d = reshape(gradient, n)

    return gradient_1d
end

function grad_q_kernel(K, y, q; C=1)
    n = length(y)

    # Convert alpha and y to 2D arrays
    q_2d = reshape(q, n, 1)
    y_2d = reshape(y, n, 1)

    # Calculate the gradient of the dual objective
    gradient = ones(n) .- (q_2d ./ y_2d) .* (y_2d .* (y_2d' * K)')
    gradient .-= (q_2d ./ y_2d)
    gradient *= C

    # Reshape gradient back to 1-D array
    gradient_1d = reshape(gradient, n)

    return gradient_1d
end

function rbf_kernel(X, Y, gamma)
    n, m = size(X, 1), size(Y, 1)
    K = zeros(n, m)

    for i in 1:n
        for j in 1:m
            norm_sq = norm(X[i, :] - Y[j, :])^2
            K[i, j] = exp(-gamma * norm_sq)
        end
    end

    return K
end
# function rbf_kernel(x, y, gamma)
#     norm_sq = norm(x - y)^2
#     return exp(-gamma * norm_sq)
# end

function grad_q(X, y, q)
    n = size(X, 1)
    m = size(X, 2)

    P = Matrix{Float64}(undef, n, n)
    for i in 1:n
        for j in 1:n
            P[i, j] = y[i] * y[j] * dot(X[i, :], X[j, :])
        end
    end

    gradient = ones(n) - P * (q .* y)
    return gradient
end


function svm_predict(X_support, y_support, alphas, b, X_test)
    n_test = size(X_test, 1)
    y_pred = Vector{Int}(undef, n_test)
    for i in 1:n_test
        decision_value = sum(alphas[j] * y_support[j] * dot(X_support[j, :], X_test[i, :]) for j in 1:length(y_support)) + b
        y_pred[i] = sign(decision_value)
    end
    return y_pred
end


function get_support_vectors(X, y, alphas; threshold=1e-7)
    # Find indices where alphas are non-zero
    sv_indices = findall(alpha -> abs(alpha) > threshold, alphas)
    sv_indices = getindex.(sv_indices, 1)

    # Extract support vectors and their labels
    X_support = X[sv_indices, :]
    y_support = y[sv_indices]

    return X_support, y_support
end

function compute_bias(X_support, y_support, alphas)
    N_SV = size(X_support, 1)
    b = 0.0
    for s in 1:N_SV
        b += y_support[s] - sum(alphas[i] * y_support[i] * dot(X_support[i, :], X_support[s, :]) for i in 1:N_SV)
    end
    return b / N_SV
end

function svm_predict(X_support, y_support, alphas, b, X_test)
    n_test = size(X_test, 1)
    y_pred = Vector{Int}(undef, n_test)
    for i in 1:n_test
        decision_value = sum(alphas[j] * y_support[j] * dot(X_support[j, :], X_test[i, :]) for j in 1:length(y_support)) + b
        y_pred[i] = sign(decision_value)
    end
    return y_pred
end

# function svm_predict_kernel(K, y_support, alphas, b)
#     n_test = size(K, 1)
#     y_pred = Vector{Int}(undef, n_test)
#     for i in 1:n_test
#         decision_value = sum(alphas[j] * y_support[j] * K[j, i] for j in 1:length(y_support)) + b
#         y_pred[i] = sign(decision_value)
#     end
#     return y_pred
# end

# function svm_predict_kernel(K, y_support, alphas, b)
#     n_test = size(K, 2)
#     decision_values = Vector{Float64}(undef, n_test)

#     for i in 1:n_test
#         for j in 1:length(y_support)
#             decision_values[i] = sum(alphas[j] * y_support[j] * K[j, i] + b)
#         end
#     end

#     y_pred = sign.(decision_values)
#     return y_pred
# end


function svm_predict_kernel(X_support, y_support, alphas, b, X_test; kernel=rbf_kernel)
    # X_support: n_support x d
    # y_support: n_support x 1
    # alphas: n_support x 1
    # b: scalar
    # X_test: n_test x d
    # kernel: function

    # compute the kernel matrix between support and test data
    # K = kernel(X_test, X_support, 0.5)

    # compute the prediction
    y_pred = zeros(size(X_test, 1))
    for i in 1:length(y_pred)
        s = 0.0
        for j in 1:length(alphas)
            s += alphas[j] * y_support[j] * (kernel(X_support[j, :]', X_test[i, :]', 0.5)[1] + b)
        end
        y_pred[i] = sign(s)
    end

    return y_pred
end

function main_GSS()
    for dataset_name in dataset_names
        trackIter = 1
        maxIter = 100
        X_train, y_train, X_test, y_test = load_and_preprocess_dataset(dataset_name)
        # define some parameters
        n = size(X_train, 1)
        d = size(X_train, 2)
        z = size(y_train, 2)

        # Dual SVM block coordinate descent with linear equality and box constraint

        # Find the initial value of alpha 
        C = 1.0
        alpha0 = solve_alpha_with_sum_constraint(C, y_train)
        # alpha0 = zeros(n, z)
        alpha0 = reshape(alpha0, :, 1)
        alpha = copy(alpha0)

        # Initialize q based on alpha
        q = alpha .* y_train
        lower = 0
        upper = C
        # gamma = 1 / length(X_train)
        gamma = 0.5
        # f(alpha) = sum(alpha[i] for i in 1:n) - 0.5 * sum(alpha[i] * alpha[j] * y[i] * y[j] * (X[i, :] ⋅ X[j, :]) for i in 1:n, j in 1:n) - lambda * sum(alpha)

        # Use the non-kernel implementation
        # f(alpha) = svm_dual_objective(X_train, y_train, alpha)
        # g(alpha) = svm_dual_gradient(X_train, y_train, alpha)
        # g_q(q) = grad_q(X_train, y_train, q)


        # use the kernel implementation
        K = rbf_kernel(X_train, X_train, gamma)
        f(alpha) = svm_dual_objective_kernel(K, y_train, alpha)
        g(alpha) = svm_dual_gradient_kernel(K, y_train, alpha)
        g_q(q) = grad_q_kernel(K, y_train, q)

        E = eigen(X_train' * X_train)
        L = maximum(E.values)
        Li = diag(X_train' * X_train)
        L2 = 2 * maximum(Li)

        # Running GSS
        alpha = copy(alpha0)
        # Initialize q based on alpha
        q = alpha .* y_train
        # use the kernel implementation
        K = rbf_kernel(X_train, X_train, gamma)
        f(alpha) = svm_dual_objective_kernel(K, y_train, alpha)
        g(alpha) = svm_dual_gradient_kernel(K, y_train, alpha)
        g_q(q) = grad_q_kernel(K, y_train, q)
        push!(gss, f(alpha))
        for i in 1:maxIter
            grad_q = g_q(q)
            alphahat, localq = GSS_project(q, grad_q, (2 * L2), lower, upper, y_train)
            alpha[:] = Float64.(alphahat[:])
            q[:] = localq[:]
            # push!(count_gss, GS1_interior)
            # push!(total_gs1, gs1_tc)

            if mod(i, trackIter) == 0
                @printf("Iteration %d, Function = %f\n", i, f(alpha))
                # push!(gs1, f(w) - fStar)
                push!(gss, f(alpha))
            end
        end
        support_vectors_GSS = findall(alpha .> 1e-6)
        # Print results
        println("Number of support vectors by GSS: ", length(support_vectors_GSS))

        # now compute the predictions
        X_support_gss, y_support_gss = get_support_vectors(X_train, y_train, alpha; threshold=1e-6)
        b = compute_bias(X_support_gss, y_support_gss, alpha)
        train_y_predicted_gss = svm_predict_kernel(X_support_gss, y_support_gss, alpha[support_vectors_GSS], b, X_train)
        correct_predictions_train_gss = sum(train_y_predicted_gss .== y_train)
        accuracy_train_gss = correct_predictions_train_gss / length(y_train)
        println("Train accuracy: ", accuracy_train_gss)

        y_predicted = svm_predict_kernel(X_support_gss, y_support_gss, alpha[support_vectors_GSS], b, X_test)
        correct_predictions_test = sum(y_predicted .== y_test)
        accuracy_test = correct_predictions_test / length(y_test)
        println("Test accuracy: ", accuracy_test)
        return gss, accuracy_train_gss, accuracy_test
    end
end
