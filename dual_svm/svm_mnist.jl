using MLJ
using Flux
using LIBSVM
using Statistics
using MLDatasets

# Load the MNIST dataset
function load_mnist()
    x_train, y_train = MLDatasets.MNIST.traindata()
    x_test, y_test = MLDatasets.MNIST.testdata()

    # Reshape images to 1D arrays and convert to Float64
    x_train = Flux.flatten(float.(x_train))'
    x_test = Flux.flatten(float.(x_test))'


    # Normalize pixel values to the range [0, 1]
    x_train = Float64.(x_train) ./ 255.0
    x_test = Float64.(x_test) ./ 255.0

    return x_train, y_train, x_test, y_test
end

x_train, y_train, x_test, y_test = load_mnist()

# Create and train the LIBSVM model
svm = LIBSVM.fit!(SVC(), x_train, y_train)

# Make predictions on the test data
y_pred = MLJ.predict(svm, x_test)

# Calculate accuracy
accuracy = mean(y_pred .== y_test)

println("Accuracy: ", accuracy)



#############################################
# using MLDatasets
# include("dual_svm_example.jl")

# # Download MNIST if needed
# # MNIST.download()

# # define train and test MNIST dataset
# X, y = MNIST.traindata()
# X_test, y_test = MNIST.testdata()

# # permute X to get the shape (60000, 784) instead of (28, 28, 60000)
# X = permutedims(X, (3, 2, 1))
# X = reshape(X, (60000, 784))
# X = X[1:5000, :]
# y = y[1:5000]

# X_test = permutedims(X_test, (3, 2, 1))
# X_test = reshape(X_test, (10000, 784))


# # solve with SCP
# n = size(X, 1)
# alpha = Variable(n)
# C = 1.0
# problem = maximize(dual_loss_DCP(alpha, X, y))
# problem.constraints += alpha >= 0
# problem.constraints += alpha <= C
# problem.constraints += sum(dot(alpha, y)) == 0

# solve!(problem, SCS.Optimizer, verbose=false, silent_solver=false)


