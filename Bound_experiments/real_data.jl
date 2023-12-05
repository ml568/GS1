function load_data(ds)
    #Life expectancy Data
    # Read the file using CSV.File and convert it to DataFrame
    if (ds == "LifeExpectancy")
        data = DataFrame(CSV.File("data/Life Expectancy Data.csv"))
        colnames = Symbol[]
        for i in string.(names(data))
            push!(colnames, Symbol(replace(replace(replace(strip(i), " " => "_"), "-" => "_"), "/" => "_")))
        end
        rename!(data, colnames)
        # Preprocess the data and select predictors and response variable
        Xtemp = [data.Adult_Mortality, data.infant_deaths, data.Alcohol, data.Hepatitis_B, data.BMI, data.GDP, data.Population]
        Xraw = Float64.(hcat(map(v -> coalesce.(v, 0), Xtemp)...))
        ytemp = [data.Life_expectancy]
        y = Float64.(hcat(map(v -> coalesce.(v, 0), ytemp)...))
        # dt = fit(ZScoreTransform, X, dims=2)
        # X = StatsBase.transform(dt, X)


    elseif (ds == "random")
        n = 1000
        d = 1000
        # Make function
        X = randn(n, d)
        w = randn(d)
        y = X * w + randn(n, 1)
        z = size(y, 2)
        return X, y


    elseif (ds == "Boston")
        # Load the Boston Housing dataset
        boston = dataset("MASS", "Boston")
        # Select predictors and response variable
        Xraw = Matrix(boston[:, 1:end-1])
        y = boston[:, :MedV]
        # dt = fit(ZScoreTransform, X, dims=2)
        # X = StatsBase.transform(dt, X)

    elseif (ds == "RB")
        data = CSV.File("data/Residential-Building-Data-Set.csv") |> DataFrame
        # Convert specific columns to Float64
        Xtemp = parse.(Float64, data[2:end, 5:end-2])
        Xraw = Matrix(Xtemp)
        # X = Matrix(data[3:end, 7:end-2])
        y = parse.(Float64, data[2:end, end-1])

    elseif (ds == "housing")
        # Load the CSV file into a DataFrame
        data = CSV.File("data/housing.csv") |> DataFrame

        # Remove rows with missing values
        data_complete = dropmissing(data)

        # Extract the features (X) and target variable (y)
        Xraw = Matrix(data_complete[:, 3:end-2])
        y = data_complete[:, :median_house_value]
    end
    n, d = size(Xraw)
    # Standardization
    X = zeros(n,d)
    for j in 1:d
        X[:,j] = (Xraw[:, j] .- mean(Xraw[:, j]))
    end
    sigma_j = std(X,dims=1)
    for j in 1:d
            X[:,j] /= sigma_j[j]
        end
    y=(y .- mean(y))./std(y)
    return X, y
end