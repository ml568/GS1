include("GS1_svm_obj.jl")
include("GSQ_svm_obj.jl")
include("GSS_svm_obj.jl")
trackIter = 1
maxIter = 100
dataset_name = "iris"

gs1, accuracy_train_gs1, accuracy_test_gs1 = main_GS1()
gss, accuracy_train_gss, accuracy_test_gss = main_GSS()
gsq, accuracy_train_gsq, accuracy_test_gsq = main_GSQ()


defs = (linewidth=5, thickness_scaling=1, xtickfontsize=16, ytickfontsize=16, legendfontsize=16)
plot!(0:trackIter:maxIter, gs1, label="GS-1", linewidth=5, thickness_scaling=1, xtickfontsize=16, ytickfontsize=16, legendfontsize=16,
    yguidefontsize=16, xguidefontsize=16, xlabel="Iteration", ylabel="f(x)")
plot!(0:trackIter:maxIter, gss, label="GS-s", linewidth=5, thickness_scaling=1, yguidefontsize=16, xguidefontsize=16, xtickfontsize=16, ytickfontsize=16, legendfontsize=16)
plot!(0:trackIter:maxIter, gsq, label="GS-q", linewidth=5, thickness_scaling=1, yguidefontsize=16, xguidefontsize=16, xtickfontsize=16, ytickfontsize=16, legendfontsize=16)
savefig("SVM_gssvsgs1gsq_wifalse_seed_" * string(seed) * dataset_name * ".pdf")

println("Train accuracy GS1: ", accuracy_train_gs1)
println("Train accuracy GSs: ", accuracy_train_gss)
println("Train accuracy GSq: ", accuracy_train_gsq)

println("Test accuracy GS1: ", accuracy_test_gs1)
println("Test accuracy GSs: ", accuracy_test_gss)
println("Test accuracy GSq: ", accuracy_test_gsq)
