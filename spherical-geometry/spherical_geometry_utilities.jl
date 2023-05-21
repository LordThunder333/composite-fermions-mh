using LinearAlgebra
import StaticPolynomials
import DynamicPolynomials
using KahanSummation
"""
Generate spinors.
"""
function u_v_generator(θ::Float64, ϕ::Float64)
    return (cos(θ) / 2 * exp(1.0im * ϕ / 2), sin(θ) / 2 * exp(-1.0im * ϕ / 2))
end
function u_v_generator(θ::Vector{Float64}, ϕ::Vector{Float64})
    return cos.(θ ./ 2) .* exp.(1.0im .* ϕ ./ 2), sin.(θ ./ 2) .* exp.(-1.0im .* ϕ ./ 2)
end
"""
Get chord distance between points on a sphere. Returns a distance matrix, the jastrow factor at p and a scale factor to normalize jastrow_derivatives.
"""
function distance_calculator(U::Vector{ComplexF64}, V::Vector{ComplexF64}, N::Int64, p::Int64)
    ### We will return a scale factor as well.
    dist_matrix::Matrix{ComplexF64} = zeros(ComplexF64, N - 1, N)
    jastrow_factor_log_times_two = zero(Float64)
    scale_factor = typemax(Float64)
    # println(typeof(scale_factor))
    for i in 1:N-1
        for j in i+1:N
            dist_matrix[j-1, i] = U[i] * V[j] - V[i] * U[j]
            if iszero(dist_matrix[j-1, i])
                return true, dist_matrix, jastrow_factor_log_times_two, scale_factor
            else
                dist_matrix[i, j] = -dist_matrix[j-1, i]
                jastrow_factor_log_times_two = jastrow_factor_log_times_two + 4 * p * log(abs(dist_matrix[i, j]))
                # println(typeof(real(dist_matrix[i,j])))
                # println(typeof(scale_factor))
                scale_factor = min(scale_factor, abs(dist_matrix[i, j]))
            end
        end
    end
    return false, dist_matrix ./ scale_factor, jastrow_factor_log_times_two, scale_factor
end

"""
Calculating f(alpha,beta) -> Park et al. for Jain-Kamilla projection.
"""
function f_α_β(α::Int64, β::Int64, U::Vector{ComplexF64}, V::Vector{ComplexF64}, dist_vector::Vector{ComplexF64})
    return sum_kbn(map(k -> V[k]^α * (-U[k])^β / dist_vector[k]^(α + β), eachindex(dist_vector)))
end

"""
Functions to generate points uniformly sampled from the surface of a sphere.
"""
function rand_θ_ϕ_gen()
    ### Generate uniform θ,ϕ from spherical surface.
    x = randn(Float64, 3)
    x = x / norm(x)
    θ = acos(x[3])
    ϕ = angle((x[1] + 1.0im * x[2]) / sin(θ))
    # if ϕ>=0
    return θ, ϕ
    # else
    #     return θ,ϕ+2π
    # end
end
function rand_θ_ϕ_gen(n_sample::Int64)
    ### Generate uniform θ,ϕ from spherical surface.
    θlist = zeros(Float64, n_sample)
    ϕlist = zeros(Float64, n_sample)
    for i in 1:n_sample
        x = randn(Float64, 3)
        x = x / norm(x)
        θ = acos(x[3])
        ϕ = angle((x[1] + 1.0im * x[2]) / sin(θ))
        # if ϕ>=0
        θlist[i], ϕlist[i] = θ, ϕ
        # else
        #      θlist[i],ϕlist[i] = θ,ϕ+2π
        # end
    end
    return map((x, y) -> (x, y), θlist, ϕlist)
end
"""
Generates jastrow derivatives for a particular electron given positions of other electrons, their distances and the jastrow polynomials and variable list.
"""
function slater_element(u::ComplexF64, v::ComplexF64, U::Vector{ComplexF64}, V::Vector{ComplexF64}, dist_vector::Vector{ComplexF64}, poly_list, var_list)
    calc_vars = [u, v]
    for var in var_list
        push!(calc_vars, f_α_β(var[1], var[2], U, V, dist_vector))
    end
    return StaticPolynomials.evaluate(poly_list, calc_vars)
end
"""
Generates derivative of jastrow factors (actually the ratio to jastrow factor)
"""
function polynomial_list_generator(qstar, p, iter_list, jastrow_derivatives)
    if qstar >= 0
        ## Parallel flux attachment. We will generate polynomial system and list of (\alpha, \beta) that are to be passed to it for computation.
        polynomial_system = Vector{DynamicPolynomials.Polynomial}(undef, size(iter_list, 1)) ### This seems simplest way to do this.
        #         var_list = []
        var_list = []
        for (n, m) in iter_list
            allowed_s_list = [s for s in 0:n if 2 * qstar + n >= qstar + n - m - s && qstar + n - m -s >= 0]
            for s in allowed_s_list
                for (key, value) in jastrow_derivatives[(s, n - s)]
                    for (var, power) in key
                        if var ∉ var_list && var ≠ (0, 0)
                            push!(var_list, var)
                        else
                            nothing
                        end
                    end
                end
            end
        end
        #         println(var_list)
        DynamicPolynomials.@polyvar u v f[1:size(var_list, 1)]
        for iter in eachindex(iter_list)
            n, m = iter_list[iter]
            ### Coeffecients of Jastrow Derivatives.
            allowed_s_list = [s for s in 0:n if 2 * qstar + n >= qstar + n - m - s && qstar + n - m -s >= 0]
            jastrow_derivatives_coeffecients = [(-1)^s * binomial(BigInt(n), BigInt(s)) * binomial(BigInt(2 * qstar + n), BigInt(qstar + n - m - s)) for s in allowed_s_list]
            jastrow_derivatives_coeffecients /= maximum(jastrow_derivatives_coeffecients) ### Will the increase accuracy? Doubtful.
            jastrow_derivatives_coeffecients = Float64.(jastrow_derivatives_coeffecients)
            ans = 0.0
            for sindex in eachindex(allowed_s_list)
                s = allowed_s_list[sindex]
                term = u^(Int(qstar + m + s)) * v^(Int(qstar - m + n - s)) * jastrow_derivatives_coeffecients[sindex]
                term_coeff = 0.0
                for (key, value) in jastrow_derivatives[(s, n - s)]
                    term1 = p^(key[(0, 0)]) * value
                    for (var, power) in key
                        if var ≠ (0, 0)
                            term1 *= f[findfirst(x -> x == var, var_list)]^(power)
                        else
                            nothing
                        end
                    end
                    term_coeff += term1
                end
                ans += term * term_coeff
            end
            #             println(ans)
            polynomial_system[iter] = DynamicPolynomials.monic(ans)
        end
        #         println(polynomial_system)
        return StaticPolynomials.PolynomialSystem(polynomial_system), var_list
    else
        ## Reverse flux attachment. We will generate polynomial system and list of (\alpha, \beta) that are to be passed to it for computation.
        polynomial_system = Vector{DynamicPolynomials.Polynomial}(undef, size(iter_list, 1)) ### This seems simplest way to do this.
        #         var_list = []
        var_list = []
        for (n, m) in iter_list
            allowed_s_list = [s for s in 0:n if 2 * abs(qstar) + n >= abs(qstar) + n - m - s && abs(qstar) + n - m -s >= 0]
            for s in allowed_s_list
                for (key, value) in jastrow_derivatives[(Int(abs(qstar) + m + s), Int(abs(qstar) - m + n - s))]
                    for (var, power) in key
                        if var ∉ var_list && var ≠ (0, 0)
                            push!(var_list, var)
                        else
                            nothing
                        end
                    end
                end
            end
        end
        #         println(var_list)
        DynamicPolynomials.@polyvar u v f[1:size(var_list, 1)]
        for iter in eachindex(iter_list)
            n, m = iter_list[iter]
            ### Coeffecients of Jastrow Derivatives.
            allowed_s_list = [s for s in 0:n if 2 * abs(qstar) + n >= abs(qstar) + n - m - s && abs(qstar) + n - m -s >= 0]
            jastrow_derivatives_coeffecients = [(-1)^s * binomial(BigInt(n), BigInt(s)) * binomial(BigInt(2 * abs(qstar) + n), BigInt(abs(qstar) + n - m - s)) for s in allowed_s_list]
            jastrow_derivatives_coeffecients /= maximum(jastrow_derivatives_coeffecients) ### Will the increase accuracy? Doubtful.
            jastrow_derivatives_coeffecients = Float64.(jastrow_derivatives_coeffecients)
            ans = 0.0
            for sindex in eachindex(allowed_s_list)
                s = allowed_s_list[sindex]
                term = u^(s) * v^(n - s) * jastrow_derivatives_coeffecients[sindex]
                term_coeff = 0.0
                for (key, value) in jastrow_derivatives[(Int(abs(qstar) + m + s), Int(abs(qstar) - m + n - s))]
                    term1 = p^(key[(0, 0)]) * value
                    for (var, power) in key
                        if var ≠ (0, 0)
                            term1 *= f[findfirst(x -> x == var, var_list)]^(power)
                        else
                            nothing
                        end
                    end
                    term_coeff += term1
                end
                ans += term * term_coeff
            end
            #             println(ans)
            polynomial_system[iter] = ans
        end
        #         println(polynomial_system)
        return StaticPolynomials.PolynomialSystem(polynomial_system), var_list
    end
end
