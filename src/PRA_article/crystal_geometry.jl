module crystal_geometry

    "=========================="
    # Preamble
    "=========================="

    using OrdinaryDiffEq, DiffEqBase, NLsolve, UnicodePlots, DrWatson;
    include(srcdir()*"/constants.jl")

    export PositionIons, Ωx, Ωy, Ωz

    "============"
    # Functions
    "============"

    "Ωx(Urf,Udc,Uec,Ωrf)"
    function Ωx(Urf,Udc,Uec,Ωrf)
       return √(1/2*(ee*Urf*αRF/(mYb*Ωrf))^2-(ee*Udc*αRODS)/mYb-(ee*Uec*αEC)/mYb)
    end

    "Ωy(Urf,Udc,Uec,Ωrf)"
    function Ωy(Urf,Udc,Uec,Ωrf)
       return √(1/2*(ee*Urf*αRF/(mYb*Ωrf))^2+(ee*Udc*αRODS)/mYb-(ee*Uec*αEC)/mYb)
    end

    "Ωz(Uec)"
    function Ωz(Uec)
       return √(2*(ee*Uec*αECz/mYb))
    end

    function diffeqs2!(ddu,du,u,p,t)
       for a in 1:Nions
          for dir in 1:3
              ddu[dir,a]=-u[dir,a]*Ω[dir]^2-cvel*du[dir,a]/mYb
          end
          for b in 1:Nions
             if a != b
                for dir in 1:3
                   ddu[dir,a]=ddu[dir,a]+(u[dir,a]-u[dir,b])*cel/((u[1,a]-u[1,b])^2 +(u[2,a]-u[2,b])^2 +(u[3,a]-u[3,b])^2)^(3/2)
                end
             end
          end
       end
    end

    function eqpos!(ddu,u)
       for a in 1:Nions
          for dir in 1:3
              ddu[dir,a]=-u[dir,a]*Ω[dir]^2
          end
          for b in 1:Nions
             if a != b
                for dir in 1:3
                   ddu[dir,a]=ddu[dir,a]+(u[dir,a]-u[dir,b])*cel/((u[1,a]-u[1,b])^2 +(u[2,a]-u[2,b])^2 +(u[3,a]-u[3,b])^2)^(3/2)
                end
             end
          end
       end
    end

    function PositionIons(Nions::Int64, ω_trap::Array{Float64,1}; plot_position::Bool=false, cvel::Float64=10E-19, tcool::Float64=100E-6, seed::Array{Float64,2}=Array{Float64}(undef, 0, 0))
         cel = ee^2/(4*π*ϵ0)/mYb;
         x0 = .4E-2;
         l0 = (ee^2/(4*π*ϵ0*mYb*ω_trap[3]^2))^(1/3);
         
         #* seed position and velocity
         if isempty(seed)
            u0 = (rand(3,Nions) .- .5)*x0;
         else
            u0 = seed;
         end

         du0 = zeros(3,Nions);

         #* system of equations
         function diffeqs2!(ddu,du,u,p,t)
            for a in 1:Nions
               for dir in 1:3
                  ddu[dir,a]=-u[dir,a]*ω_trap[dir]^2-cvel*du[dir,a]/mYb
               end
               for b in 1:Nions
                  if a != b
                     for dir in 1:3
                        ddu[dir,a]=ddu[dir,a]+(u[dir,a]-u[dir,b])*cel/((u[1,a]-u[1,b])^2 +(u[2,a]-u[2,b])^2 +(u[3,a]-u[3,b])^2)^(3/2)
                     end
                  end
               end
            end
         end

         function eqpos!(ddu,u)
            for a in 1:Nions
               for dir in 1:3
                  ddu[dir,a]=-u[dir,a]*ω_trap[dir]^2
               end
               for b in 1:Nions
                  if a != b
                     for dir in 1:3
                        ddu[dir,a]=ddu[dir,a]+(u[dir,a]-u[dir,b])*cel/((u[1,a]-u[1,b])^2 +(u[2,a]-u[2,b])^2 +(u[3,a]-u[3,b])^2)^(3/2)
                     end
                  end
               end
            end
         end

         #* solve numerically
         tspan = (0.0,tcool);
         prob = SecondOrderODEProblem(diffeqs2!,du0,u0,tspan);
         sol_approx = solve(prob,Tsit5(),save_everystep=false);
         u_min = reshape(sol_approx[3*Nions+1:end, end],3, Nions)
         
         sol_exact = nlsolve(eqpos!, u_min, autodiff = :forward)
         
         x=sol_exact.zero[1,1:end].*1E6; 
         y=sol_exact.zero[2,1:end].*1E6; 
         z=sol_exact.zero[3,1:end].*1E6;

         #* plot positions using UnicodePlots
         x_max = maximum(x)+5;x_min = minimum(x)-5;
         y_max = maximum(y)+5;y_min = minimum(y)-5;
         z_max = maximum(z)+5;z_min = minimum(z)-5;

         plot_position == true && return (sol_exact.zero)/l0, [scatterplot(x,z;border=:corners,grid=:false,xlabel="X [μm]",ylabel="Z [μm]",xlim=(x_min,x_max),ylim=(z_min,z_max)),  scatterplot(y,z;border=:corners,grid=:false,xlabel="Y [μm]",ylabel="Z [μm]",xlim=(y_min,y_max),ylim=(z_min,z_max))]
         
         plot_position == false && return (sol_exact.zero)/l0
    end

    function PositionIons(Nions::Int64, ωtrap::Array{Float64,1}, ion_distance::Float64, coeffs::Array{Float64}; plot_position::Bool=false, cvel::Float64=3E-20, tcool::Float64=200E-6)
      Ω=ωtrap;
      cel=ee^2/(4*π*ϵ0*mYb);
      size_crystal = ion_distance*Nions
      u0=size_crystal*rand(-1.0:1e-6:1.0,(3,Nions)); du0=zeros(3,Nions); tspan = (0.0,tcool);
      
      ### Field paul trap
      a=coeffs[1]; b=coeffs[2];
      Ez(x,y,z) = -(a*z + b*z^3 - (3b/2)*z*(x^2 + y^2))/ee;
      Ex(x,y,z) = -(-a/2*x - (3b/2)*z^2*x + (3b/4)*(x^3/3+y^2*x))/ee;
      Ey(x,y,z) = -(-a/2*y - (3b/2)*z^2*y + (3b/4)*(x^2*y+y^3/3))/ee;
      Etrap = [Ex,Ey,Ez]
      
      function approxpos!(ddu,du,u,p,t) #remove p and t input variables
         for α in 1:3, i in 1:Nions
            ddu[α,i]= -(α!=3)*u[α,i]*Ω[α]^2 - (ee/(mYb))*Etrap[α](u[:,i]...) - cvel*du[α,i]/mYb #Paul trap + Cooling force
            for j in 1:Nions
               i != j && (ddu[α,i] = ddu[α,i] + cel*(u[α,i]-u[α,j])/(sum((u[:,i]-u[:,j]).^2)^1.5)) #Coulomb force
            end
         end
      end
     
      function exactpos!(ddu,u)
         for α in 1:3, i in 1:Nions
            ddu[α,i]= -(α!=3)*u[α,i]*Ω[α]^2 - (ee/mYb)*Etrap[α](u[:,i]...) #Paul trap
            for j in 1:Nions
               i != j && (ddu[α,i] = ddu[α,i] + cel*(u[α,i]-u[α,j])/(sum((u[:,i]-u[:,j]).^2)^1.5)) #Coulomb force
            end
         end
      end


      prob = SecondOrderODEProblem(approxpos!,du0,u0,tspan);
      sol_approx = solve(prob,Tsit5(),save_everystep=false);
      u_min = reshape(sol_approx[3*Nions+1:end, end],3, Nions)
      sol_exact = nlsolve(exactpos!, u_min, autodiff = :forward)
      x=sol_exact.zero[1,1:end].*1E6; y=sol_exact.zero[2,1:end].*1E6; z=sol_exact.zero[3,1:end].*1E6;
      x_max = maximum(x)+5;x_min = minimum(x)-5;y_max = maximum(y)+5;y_min = minimum(y)-5;z_max = maximum(z)+5;z_min = minimum(z)-5;
      plot_position == true && return (sol_exact.zero)/ion_distance, [scatterplot(x,z;border=:corners,grid=:false,xlabel="X [μm]",ylabel="Z [μm]",xlim=(x_min,x_max),ylim=(z_min,z_max)),  scatterplot(y,z;border=:corners,grid=:false,xlabel="Y [μm]",ylabel="Z [μm]",xlim=(y_min,y_max),ylim=(z_min,z_max))]
      plot_position == false && return (sol_exact.zero)/ion_distance
    end
    
    function TrapPotential(x0,y0,z0)
      x, y, z = Sym("x y z"); a, b, e = Sym("a b e");
      Ez(x,y,z) = -(a*z + b*z^3 - (3b/2)*z*(x^2 + y^2))/e;
      Ex(x,y,z) = -(-a/2*x - (3b/2)*z^2*x + (3b/4)*(x^3/3+y^2*x))/e;
      Ey(x,y,z) = -(-a/2*y - (3b/2)*z^2*y + (3b/4)*(x^2*y+y^3/3))/e;
      ∫Ex = SymPy.integrate(Ex(x,0,0),x)
      ∫Ey = SymPy.integrate(Ey(x,y,0),y)
      ∫Ez = SymPy.integrate(Ez(x,y,z),z)
      Vx = ∫Ex-∫Ex(x=>0)
      Vy = ∫Ey-∫Ey(y=>0)
      Vz = ∫Ez-∫Ez(z=>0)
      Vt = Vx + Vy + Vz
      return Vt(x=>x0, y=>y0, z=>z0)
      #Or the full-expression: Vt(x,y,z) = a*x^2/(4*e) - b*x^4/(16*e) - b*y^4/(16*e) - b*z^4/(4*e) + y^2*(2*a - 3*b*x^2)/(8*e) + z^2*(-2*a + 3*b*x^2 + 3*b*y^2)/(4*e)
    end
	

end  # module crystal_geometry


"======"
# Old
"======"

   function SecondDerivative()
      dir = [Sym("x"),Sym("y"),Sym("z")];
      Vtrap(x,y,z) = a*x^2/(4*e) - b*x^4/(16*e) - b*y^4/(16*e) - b*z^4/(4*e) + y^2*(2*a - 3*b*x^2)/(8*e) + z^2*(-2*a + 3*b*x^2 + 3*b*y^2)/(4*e)
      ddVtrap = [diff(Vtrap(x,y,z), dir[α], dir[β]) for α=1:3, β=1:3]
   end

