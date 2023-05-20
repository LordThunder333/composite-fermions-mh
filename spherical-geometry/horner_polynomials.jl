### We follows Park et al. to calculate the ratio of mixed derivatives of jastrow factors (raised to some integer power p) with respect to spinor variables u and v to the jastrow factor.
jastrow_derivatives = Dict()
jastrow_derivatives[(0,0)] = Dict(Dict((0,0)=>0)=>one(ComplexF64))

### J.26 in book.
function opV(powers, coeff::ComplexF64)
    deriv_dict = Dict()
    for (key, value) in powers
#         println(key)
        working_dict = copy(powers)
        if key == (0,0)
            working_dict[(0,0)] += 1
            if haskey(powers, (0,1))
                working_dict[(0,1)] += 1
            else
                working_dict[(0,1)] = 1
            end
            if haskey(deriv_dict, working_dict)
                deriv_dict[working_dict] += coeff
            else
                deriv_dict[working_dict] = coeff
            end
        elseif value==0
            nothing
        else
            α,β = key
            working_dict[(α,β)] -= 1
            if working_dict[(α,β)]==0
                delete!(working_dict,(α,β))
            else
               nothing 
            end
            if haskey(working_dict,(α,β+1))
                    working_dict[(α,β+1)] +=1
            else
                    working_dict[(α,β+1)] =1
            end
            if haskey(deriv_dict, working_dict)
                deriv_dict[working_dict] += coeff*(-(α+β))*value
            else
                deriv_dict[working_dict] = coeff*(-(α+β))*value
            end
        end        
    end
    return deriv_dict
end

### J.27 in book.
function opU(powers, coeff::ComplexF64)
    deriv_dict = Dict()
    for (key, value) in powers
#         println(key)
        working_dict = copy(powers)
        if key == (0,0)
            working_dict[(0,0)] += 1
            if haskey(powers, (1,0))
                working_dict[(1,0)] += 1
            else
                working_dict[(1,0)] = 1
            end
#             println(working_dict)
            if haskey(deriv_dict, working_dict)
                deriv_dict[working_dict] += coeff
            else
                deriv_dict[working_dict] = coeff
            end
        elseif value==0
            nothing
        else
            α,β = key
            working_dict[(α,β)] -= 1
            if haskey(working_dict,(α+1,β))
                    working_dict[(α+1,β)] +=1
            else
                    working_dict[(α+1,β)] =1
            end
            if working_dict[(α,β)]==0
                delete!(working_dict,(α,β))
            else
               nothing 
            end

            if haskey(deriv_dict, working_dict)
                deriv_dict[working_dict] += coeff*(-(α+β))*value
            else
                deriv_dict[working_dict] = coeff*(-(α+β))*value
            end
        end        
    end
    return deriv_dict
end

### Adding up operator actions.
function dict_combine(dict_list)
    ans_dict = Dict()
    for each in dict_list
        for (key, value) in each
                if haskey(ans_dict, key)
                    ans_dict[key] += value
                else
                    ans_dict[key] = value
                end
        end
    end
    return ans_dict
end

###
for v in 1:10
   jastrow_derivatives[(0,v)] = dict_combine([opV(powers,coeff) for (powers,coeff) in jastrow_derivatives[(0,v-1)]])
end
for v in 0:10
    for u in 1:10
        jastrow_derivatives[(u,v)] =  dict_combine([opU(powers,coeff) for (powers,coeff) in jastrow_derivatives[(u-1,v)]])
    end
end
deserialize("./jastrow_derivatives.jls")
