using JuMP
using GLPK

function solve_alpha_with_sum_constraint(u, y)
    model = Model(GLPK.Optimizer)
    n = length(y)

    @variable(model, alpha[1:n])

    for i in 1:n
        set_lower_bound(alpha[i], 0.1)
        set_upper_bound(alpha[i], u)
    end

    # Add the constraint for sum(q_i) = 0
    @constraint(model, alpha' * y == 0)
    # @constraint(model, alpha .> 0)


    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        return value.(alpha)
    else
        error("No optimal solution found.")
    end
end

# # Example usage:
# u = 1
# # alpha = [0.5, 0.5, 0.5, 0.5]
# y = [1, -1, 1, 1, -1, 1]

# alpha = solve_q_with_sum_constraint(u, y)
# println(alpha)
# print(alpha' * y)
