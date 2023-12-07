function load_and_preprocess_dataset(dataset_name)
    if dataset_name == "iris"
        dataset_obj = dataset("datasets", dataset_name)
        dataset_df = DataFrame(dataset_obj)
        # Select two classes for binary classification, e.g., "Setosa" and "Versicolor"
        class1 = "setosa"
        class2 = "versicolor"
        binary_df = vcat(dataset_df[dataset_df[:, :Species].==class1, :], dataset_df[dataset_df[:, :Species].==class2, :])
        binary_df = binary_df[shuffle(1:end), :]
        # Split the binary dataset into training and testing sets as before
        train_percentage = 0.7
        train_size = Int(round(train_percentage * size(binary_df, 1)))
        X_train = Matrix(binary_df[1:train_size, 1:4])  # Features
        y_train = Vector(binary_df[1:train_size, :Species] .== class1)  # Binary labels: Setosa (true) or Versicolor (false)
        X_test = Matrix(binary_df[train_size+1:end, 1:4])
        y_test = Vector(binary_df[train_size+1:end, :Species] .== class1)
    end
    dt = fit(ZScoreTransform, X_train, dims=2)
    X_train = StatsBase.transform(dt, X_train)

    dt = fit(ZScoreTransform, X_test, dims=2)
    X_test = StatsBase.transform(dt, X_test)
    y_train = Int.(y_train)
    y_test = Int.(y_test)
    y_train[y_train.==0] .= -1
    y_test[y_test.==0] .= -1

    return X_train, y_train, X_test, y_test

end