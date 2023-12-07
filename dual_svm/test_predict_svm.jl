using RDatasets
using LIBSVM
using Random
using LinearAlgebra


function rbf_kernel(X, Y, gamma)
    # X and Y are both NxD matrices
    # gamma is the kernel parameter
    # returns a NxN matrix of kernel evaluations
    N = size(X, 1)
    K = zeros(N, N)
    for i in 1:N
        for j in 1:N
            K[i, j] = exp(-gamma * norm(X[i, :] - Y[j, :]))
        end
    end
    return K
end

function linear_kernel(X, Y)
    # Implements the linear kernel for SVM. X and Y are both 
    # NxD matrices
    N = size(X, 1)
    K = zeros(N, N)
    for i in 1:N
        for j in 1:N
            K[i, j] = sum(X[i, :] .* Y[j, :])
        end
    end
    return K
end

# function rbf_kernel(X, Y, gamma)
#     n, m = size(X, 1), size(Y, 1)
#     K = zeros(n, m)

#     for i in 1:n
#         for j in 1:m
#             norm_sq = norm(X[i, :] - Y[j, :])^2
#             K[i, j] = exp(-gamma * norm_sq)
#         end
#     end

#     return K
# end

# function svm_predict_kernel(X_support, y_support, alphas, b, X_test)

#     K = rbf_kernel(X_support, X_test, 0.5)

#     n_test = size(K, 2)
#     decision_values = Vector{Float64}(undef, n_test)

#     for i in 1:n_test
#         for j in 1:length(y_support)
#             decision_values[i] = sum(alphas[j] * y_support[j] * K[j, i] + b)
#         end
#     end

#     y_pred = sign.(decision_values)
#     return y_pred, decision_values
# end

function svm_predict(X_support, y_support, alphas, b, X_test; kernel=rbf_kernel)
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
            s += alphas[j] * (kernel(X_support[j, :]', X_test[i, :]')[1] + b)
        end
        y_pred[i] = sign(s)
    end

    return y_pred
end


iris = dataset("datasets", "iris")

Random.seed!(123)  # For reproducibility
# Load the Iris dataset
iris = dataset("datasets", "iris")

# Convert the dataset to a DataFrame for easier manipulation
iris_df = DataFrame(iris)

# Select two classes for binary classification, e.g., "Setosa" and "Versicolor"
class1 = "setosa"
class2 = "versicolor"

# Create a binary classification dataset
binary_df = vcat(iris_df[iris_df[:, :Species].==class1, :], iris_df[iris_df[:, :Species].==class2, :])

# Shuffle the binary dataset
using Random
binary_df = binary_df[shuffle(1:end), :]

# Split the binary dataset into training and testing sets as before
train_percentage = 0.7
train_size = Int(round(train_percentage * size(binary_df, 1)))

X_train = Matrix(binary_df[1:train_size, 1:4])  # Features
y_train = Vector(binary_df[1:train_size, :Species] .== class1)  # Binary labels: Setosa (true) or Versicolor (false)
X_test = Matrix(binary_df[train_size+1:end, 1:4])
y_test = Vector(binary_df[train_size+1:end, :Species] .== class1)

y_train = Int.(y_train)
y_test = Int.(y_test)

# Convert negative labels to -1
y_train[y_train.==0] .= -1
y_test[y_test.==0] .= -1


# Example: Using a RBF kernel
model = svmtrain(X_train', y_train, kernel=LIBSVM.Kernel.Linear)
y_pred, decision_values = svmpredict(model, X_test')
accuracy = sum(y_pred .== y_test) / length(y_test)
println("Accuracy: $accuracy")



# Get the support vectors and other required parameters
# alphas = model.SVs_coef
# alphas = model.coefs
b = model.rho[1]  # Assuming it's a one-class SVM, otherwise, you might need to handle multiple classes

# Get the support vectors themselves
sv_indices = model.SVs.indices
X_support = X_train[sv_indices, :]
y_support = y_train[sv_indices]
# alphas = find_alphas(y_train, X_support, b, X_train)
alphas = model.coefs
# alphas = abs.(alphas)

# Call your svm_predict function
y_pred = svm_predict(X_support, y_support, alphas, b, X_test; kernel=linear_kernel)
# y_pred = sign.(decisions)

# Calculate the accuracy of your predictions
accuracy = sum(y_pred .== y_test) / length(y_test)

println("Accuracy: $accuracy")

