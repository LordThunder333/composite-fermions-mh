import Base.*
import FileIO
import JLD2
using LinearAlgebra
"""
System Object.
"""
mutable struct system{T}
    state_vector::Vector{T} ### Current state.
    logpdf::Float64 ### Log of current state vector.
    # amp_vector::Vector{ComplexF64} ### List of amplitudes.
    # poly_main ### Polynomial system corresponding to the sampling distribution.
    # var_list_main ### Variable list corresponding to the sampling distribution.
    # poly_super ### Superset of polynomials
    # var_list_super ### Superset of var_list
    # index_list ### List of indices.
    params::Vector{Any}
end

"""
Logpdf of the system with the vector of parameters passed in the same order as arguments.
"""
function logpdf(ψ::system{T}) where {T}
    # return logpdf(ψ.state_vector::Vector{T}, ψ.poly_main, ψ.var_list_main, ψ.poly_other,ψ.var_list_other,ψ.index_list,Tuple(ψ.params)...)
    return logpdf(ψ.state_vector::Vector{T}, Tuple(ψ.params)...)
end

"""
Samples object containing all metropolis-hastings samples, their logpdfs, acceptance ratio and amplitudes list for the samples, system size and number of samples. Two samples object can be joined using *.
"""
mutable struct samples{T}
    sample_vector::Matrix{T}
    logpdf_vector::Vector{Float64}
    # amp_vector::Matrix{ComplexF64}
    acceptance_ratio::Float64
    N_system::Int64 ### System size.
    N_samples::Int64 ### Number of samples.
end
"""
Function to save samples object in a compressed format.
"""
function save_samples(samples_current::samples, str::String)
    FileIO.save(str*".jld2",Dict("sample_vector"=>samples_current.sample_vector, "logpdf_vector"=>samples_current.logpdf_vector,"acceptance_ratio"=>samples_current.acceptance_ratio,"N_system"=>samples_current.N_system,"N_samples"=>samples_current.N_samples))
    return
end
"""
Function to load samples object from a compressed format.
"""
function load_samples(str::String)
    return samples(FileIO.load(str*".jld2","sample_vector","logpdf_vector","acceptance_ratio","N_system","N_samples")...)
end
"""
Function to join two samples object.
"""
function *(a::samples{T}, b::samples{T}) where T
    # println(T)
    N_system::Int64 = a.N_system
    N_samples::Int64 = a.N_samples + b.N_samples
    # sample_vector = Array{T, 2}(undef,N_system,N_samples)
    sample_vector = hcat(a.sample_vector, b.sample_vector)
    # logpdf_vector = Array{Float64, 1}(undef,N_samples)
    logpdf_vector = vcat(a.logpdf_vector,b.logpdf_vector)
    # amp_vector = hcat(a.amp_vector,b.amp_vector)
    # sample_vector[:,begin:a.N_samples] = copy(a.sample_vector)
    # sample_vector[:,a.N_samples+1:end] = copy(b.sample_vector)
    # logpdf_vector[begin:a.N_samples] = copy(a.logpdf_vector)
    # logpdf_vector[a.N_samples+1:end] = copy(b.logpdf_vector)
    return samples(sample_vector, logpdf_vector,(a.N_samples * a.acceptance_ratio + b.N_samples * b.acceptance_ratio)/N_samples, N_system, N_samples)
end
"""
Primary workhorse of metropolis-hastings sampling paradigm. Takes a system object, alongwith a proposal distribution and related parameters. Modifies system object.
"""
function mh_sampler!(ψ::system{T}, q::Function, q_params...) where T
    X = ψ.state_vector
    Y = q(X, q_params...)
    logpdf_Y = logpdf(Y, Tuple(ψ.params)...)
    if logpdf_Y-ψ.logpdf >= log(rand())
        ψ.state_vector = copy(Y)
        # ψ.amp_vector = copy(amp_vector_Y)
        ψ.logpdf = logpdf_Y
        return true
    else
        return false
    end
end
"""
Batch sampler for metropolis hastings. Returns a sample object.
"""
function mh_batch_sampler!(batch_size::Int64, ψ::system{T}, q::Function, q_params...) where T
    ans::samples{T} = samples(Matrix{T}(undef, size(ψ.state_vector,1), batch_size), Vector{Float64}(undef, batch_size), zero(Float64), size(ψ.state_vector,1), batch_size)
    n_accepted::Int64 = zero(Int64)
    for i in 1:batch_size
        n_accepted += Int64(mh_sampler!(ψ, q, q_params...))
        ans.sample_vector[:,i] = copy(ψ.state_vector)
        ans.logpdf_vector[i] = ψ.logpdf
        # ans.amp_vector[:,i] = copy(ψ.amp_vector)
    end
    ans.acceptance_ratio::Float64 = n_accepted/batch_size
    return ans
end
"""
Simulated annealing sampler. Takes system object, temperature, proposal function and related parameters as arguments. Modifies system object.
"""
function simulated_annealing_sampler!(ψ::system, temp::Float64, q::Function, q_params...)
    X = ψ.state_vector
    Y = q(X, q_params...)
    logpdf_Y = logpdf(Y, Tuple(ψ.params)...)
    if logpdf_Y-ψ.logpdf > 0 || logpdf_Y-ψ.logpdf > temp*log(rand())
        ψ.state_vector = copy(Y)
        # ψ.amp_vector = copy(amp_vector_Y)
        ψ.logpdf = logpdf_Y
        return true
    else
        return false
    end
    # return
end
"""
Simulated annealing batch sampler. Takes batch size, temperature for each sample, system object, proposal function and related parameters as arguments. Modifies system object.
"""
function simulated_annealing_batch_sampler!(batch_size::Int64, temp_vector::Vector{Float64}, ψ::system{T}, q::Function, q_params...) where T
    for i in 1:batch_size
        simulated_annealing_sampler!(ψ, temp_vector[i], q, q_params...)
    end
    return
end

"""
Returns parameters for ARM scheme for step size adapation to maintain acceptance ratio.
"""
function arm_parameters(ideal_acceptance_ratio::Float64, r::Float64)
    a = 1.0
    b = 0.0
    for i in 1:1000
        c = (a*ideal_acceptance_ratio+b)^r
        a = (a*ideal_acceptance_ratio+b)^(1/r) - c
        b = c
    end
    return a, b
end
"""
Returns ARM scale factor for a given acceptance ratio, the ideal acceptance ratio and ARM parameters.
"""
function arm_scale_factor(p, p_i, a, b)
    return log(a*p_i + b)/log(a*p + b)
end
