module models

    module general
       
        using LinearAlgebra
        
        export NearestNeighbour

        "Calculates nearest neighbours"
        function NearestNeighbour(positions, cutoff)
            number_ions = size(positions)[2]
            indicesNN = []
            for m=1:number_ions
                dₘ = [norm(positions[:,m]-positions[:,n]) for n=1:number_ions]
                indexNN = findall(d -> d < cutoff && d != 0, dₘ)
                push!(indicesNN, indexNN)
            end
            return indicesNN
        end
    end

    module Kondo

        using LinearAlgebra

        export TriangularLattice

        "Generates evenly spaced triangular lattice plaqutte. Depth indicates the distance to the central ion"
        function TriangularLattice(d₀, depth)
            d₁ = d₀*cos(π/6)
            
            function DisplaceYZ(unit_cell)
                left_cell = unit_cell + [0 0 0; -d₀/2 -d₀/2 -d₀/2; d₁ d₁ d₁]
                right_cell = unit_cell + [0 0 0; d₀/2 d₀/2 d₀/2; d₁ d₁ d₁]
                return left_cell, right_cell
            end

            u0 = [0 0 0; 0 -d₀/2 d₁; 0 d₀/2 d₁]' #first unit cell depth 0

            #* build (north) unit cell up to target depth
            ud = []
            push!(ud, u0)
            for d = 0:depth-2
                for i = length(ud)-d:length(ud)
                    uleft, uright = DisplaceYZ(ud[i])
                    push!(ud, uleft)
                    push!(ud, uright)
                end
                unique!(ud)
            end
            ud_a = unique(hcat(ud...),dims=2)

            #* Rotate unit cell to obtain complete crystal

            R(n)=[0 0 0; 0 cos(n*π/3) sin(n*π/3); 0 cos(n*π/3+π/2) sin(n*π/3+π/2)] #Rotation matrix

            ut = []
            for n=0:5
                push!(ut, R(n)*ud_a)
            end

            #* filter repeated coordinates
            positions_ions = replace(round.(hcat(ut...), digits=6), -0.0 => 0.0) #some ridiculous filtering
            positions_ions = unique(positions_ions, dims=2)

            #* sort by distance and angle
            positions_sorted = sortslices(positions_ions, dims=2, by = x -> norm(x))
            
            for d in 1:depth
                total_num_vertices = 6*Int(d*(d+1)/2)+1;
                first_index_level = total_num_vertices - 6*d + 1;
                positions_sorted[:,first_index_level:total_num_vertices] = sortslices(positions_sorted[:,first_index_level:total_num_vertices], dims=2, by = x -> atan(x[2],x[3]))
            end
            return positions_sorted
        end

        function TriangularLattice(d₀)
            triang_lattice = zeros(3,19)
    
            d₁ = d₀*cos(π/6)
            for n=1:6
                triang_lattice[:,n+1] = [0.0, d₀*cos(n*π/3), d₀*sin(n*π/3)] #depth=1
            end
            for n=1:6
                triang_lattice[:,2*n+7] = [0.0, 2*d₀*cos(n*π/3), 2*d₀*sin(n*π/3)] #depth =2; vertices
            end
            for n=1:6
                triang_lattice[:,2*n-1+7] = [0.0, 2*d₁*cos(n*π/3-π/6), 2*d₁*sin(n*π/3-π/6)] #depth =2; middle points
            end
    
    
            return triang_lattice
        end


    end
end