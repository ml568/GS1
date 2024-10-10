maxIter = 50 # 50k in paper

# Synthetic
seeds = [30,50,1,2,3]
dataset = "syn1"
include("main_equality.jl")
dataset = "syn2"
include("main_equality.jl")

seeds = [0]

# Normal
dataset = "australian_scale"
include("main_equality.jl")
dataset = "ionosphere"
include("main_equality.jl")
dataset = "sonar"
include("main_equality.jl")
dataset = "german"
include("main_equality.jl")
dataset = "svmguide1"
include("main_equality.jl")
dataset = "svmguide3"
include("main_equality.jl")
dataset = "splice"
include("main_equality.jl")
dataset = "ijcnn1"
include("main_equality.jl")

# Weird
#dataset = "fourclass_scale" # Removed since it only has 2 variables
#include("main_equality.jl")
dataset = "colon-cancer"
include("main_equality.jl")
dataset = "duke-breast-cancer"
include("main_equality.jl")
dataset = "mushrooms"
include("main_equality.jl")
dataset = "madelonStd"
include("main_equality.jl")
dataset = "phishing"
include("main_equality.jl")
dataset = "a1a"
include("main_equality.jl")
dataset = "w1a"
include("main_equality.jl")
