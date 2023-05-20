using Serialization
# using ThreadPinning
# pinthreads(:cores)
include("spherical_geometry_utilities.jl")
include("monte_carlo_utilities.jl")

global const jastrow_derivatives = deserialize("./jastrow_derivatives.jls")

"""
Function which calculates the log of pdfs: returns log(ψ1^2). Takes in a position vector, Qstar (effective magnetic field strength faced by CFs), p (CF flavour), StaticPolynomials system - Contains all the jastrow derivatives for various CF orbitals, list of f(alpha,beta) to be calculated,list of orbitals given as (Lambda Level Index, Lz).
"""
function logpdf(θ_ϕ::Vector{Tuple{Float64, Float64}}, Qstar, p::Int64, poly_list, var_list, iter_list)
    θ = [x[1] for x in θ_ϕ] ### Slightly ineffecient. But such is life.
    ϕ = [x[2] for x in θ_ϕ]
    U, V = u_v_generator(θ,ϕ)
    electrons_together::Bool, dist_matrix, jastrow_factor_log_times_two, scale_factor = distance_calculator(U, V, size(θ, 1), p)
    if electrons_together
        return -Inf64
    # elseif n==1
    #     return jastrow_factor_log_times_two * (2*p + 1)/(2*p) + sum_kbn(map(x->log(sin(x)),θ))
    else
        slater_det = zeros(ComplexF64, size(θ, 1), size(θ, 1))
      
        ### Can be multithreaded if necessary.
        for electron_iter in 1:size(θ, 1)
             slater_det[:,electron_iter] = slater_element(U[electron_iter],V[electron_iter],U[begin:end .!= electron_iter],V[begin:end .!= electron_iter],dist_matrix[:,electron_iter],poly_list,var_list)
        end
        if Qstar >=0
            return jastrow_factor_log_times_two + 2*real(logdet(slater_det)) - 2*log(scale_factor)*sum(x->x[1], iter_list) + sum_kbn(map(x->log(sin(x)),θ))
        else
            return jastrow_factor_log_times_two + 2*real(logdet(slater_det)) - 2*log(scale_factor)*sum(x->x[1] + 2*abs(Qprime), iter_list) + sum_kbn(map(x->log(sin(x)),θ))
        end
    end
end

"""
Generic Monte Carlo Sampler. Below implements a CF ground state samples using Gibbs sampling proposal.
"""
function monte_carlo_sampler_gs(chain_number::Int64,n_batches::Int64, batch_size::Int64,N::Int64,n::Int64,p::Int64)
    ### N-1 electron in n levels such that there is one hole. n*(2Qstar + 1) + 2*(n-1)*n/2 = n^2 + 2*Qstar*n = 
    Qstar = (N//n-n)//2### Qstar.
    iter_list = [(ll_index,m) for ll_index in 0:n-1 for m in -(Qstar+ll_index):1:(Qstar+ll_index)]

    initial_sample = rand_θ_ϕ_gen(N) ### Number of electrons in N-1.
    poly_list, var_list = polynomial_list_generator(Qstar, p, iter_list, jastrow_derivatives)

    global gibbs_iter = 1

    function proposal(Z::Vector{Tuple{Float64, Float64}}, σθ::Float64, σϕ::Float64, N1::Int64) ### Here, we have sample vector.
        Z0 = copy(Z)
#         dtheta = randn(σθ)
        # for i in 1:N1

        # end
        if gibbs_iter==N1
            iter = N1
            gibbs_iter = 1
        else
            iter = gibbs_iter
            gibbs_iter += 1
        end
        # θnew = map(x->x[1], Z0) .+ σθ*randn(Float64,N1)
        # θnew = rand(MvNormal(map(x->x[1], Z), σθ))
        # ϕnew = rand(MvNormal(map(x->x[2], Z), σϕ))
        θ = Z0[iter][1] + σθ*randn()
        ϕ = Z0[iter][2] + σϕ*randn()
        # θproper = acos.(cos.(θnew))
        # ϕproper = angle.( (sin.(θnew) .* cos.(ϕnew) .+ 1.0im .* sin.(θnew) .* sin.(ϕnew)) ./ sin.(θproper))
        x = [sin(θ)*cos(ϕ), sin(θ)*sin(ϕ),cos(θ)]
        θ = acos(x[3])
        ϕ = angle((x[1]+1.0im*x[2])/sin(θ))
        Z0[iter] = (θ,ϕ)
        return Z0
        # return map(i->(θproper[i],ϕproper[i]),1:N1)
end

        ### We have proposal distribution.
    # σθ = Float64(π) .* Matrix{Float64}(I, N-1, N-1)
    # σϕ = Float64(2*π) .* Matrix{Float64}(I, N-1, N-1) 
  
    ### This requires some rethinking. What is the ideal acceptance ratio for thermalization??
    σθ = Float64(π)
    σϕ = Float64(2*π)

    initial_logpdf = logpdf(initial_sample, Qstar, p, poly_list, var_list, iter_list)
    ψ = system(initial_sample, initial_logpdf, [Qstar, p, poly_list, var_list, iter_list])
    
    ### Simulated annealing.
    simulated_annealing_batch_sampler!(round(Int64, div(n_batches*batch_size,8)),[10^(4*(Int64(div(n_batches*batch_size,8))-i)/(Int64(div(n_batches*batch_size,8))-1)) for i in 1:Int64(div(n_batches*batch_size,8))],ψ,proposal,σθ,σϕ,N)

    # σθ = 1e-4 * Float64(π)/(N-1) .* Matrix{Float64}(I, N-1, N-1)
    σθ = Float64(π)/(N)
    # σϕ = 1e-4 * Float64(2*π)/(N-1) .* Matrix{Float64}(I, N-1, N-1)
    σϕ = Float64(2*π)/(N)

    pideal = 0.50
    a, b = arm_parameters(pideal, 3.0)
    δ = 1.0

    samples_current = mh_batch_sampler!(batch_size, ψ, proposal, σθ, σϕ, N)
    save_samples(samples_current,"samples_current_cf_gs-chain-$(chain_number)-$N-$n-$(2*n*p+1)-post-thermalization")
    for iter in 1:n_batches-1
        # σθ = cov(map(x->x[1],samples_current.sample_vector); dims=2)
        σθ = mean(diag(cov(map(x->x[1],samples_current.sample_vector); dims=2)))
        σϕ = mean(diag(cov(map(x->x[2],samples_current.sample_vector); dims=2)))
        # σϕ = cov(map(x->x[2],samples_current.sample_vector); dims=2)
        δ = δ*arm_scale_factor(samples_current.acceptance_ratio, pideal, a, b)
        samples_current =  samples_current*mh_batch_sampler!(batch_size, ψ, proposal, δ * σθ, δ * σϕ, N)
    end
    save_samples(samples_current,"samples_current_cf_gs-chain-$(chain_number)-$N-$n-$(2*n*p+1)-final")
    return
end
