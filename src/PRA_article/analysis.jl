module analysis

    export CouplingErrors

    #* calculate absolute errors 
    function CouplingErrors(target_matrix, result_matrix)
        dims = size(target_matrix)[2]
        target_couplings = result_matrix - result_matrix.*(iszero.(target_matrix))
        residual_couplings = result_matrix.*(iszero.(target_matrix))
        error_target_couplings = [target_couplings[i,j]!=0 ? (target_matrix[i,j]-target_couplings[i,j]) : 0 for i in 1:dims, j in 1:dims]
        return error_target_couplings, residual_couplings
    end

end