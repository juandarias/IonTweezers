Nions=7
JPL(α) = [(i!=j)*abs(i-j)^(-Float64(α)) for i=1:Nions, j=1:Nions]

λm, xm= eigen(JPL(1))



bb =sum(xm[:,n]*xm[:,n]' for n=1:Nions)
λm1, xm1 = eigen(bb)

xm1 - xm

bar(1:10,xm[:,10], width=0.1); gcf()
plt.figure()
for n=1:4
    cc = 1
    λm, xm= eigen(JPL(n))
    n==1 && (cc = -1)
    bar(collect(1:7).+0.1*(n-1), cc*xm[:,1], width=0.1, label="α="*string(n));
end
xlabel("Ion");ylabel("Amplitude");
title("Amplitudes of \$\\vec b_{min}\$")
legend()
gcf()



ratio_ev = []
plt.figure()
for n=1:4
    λm, xm= eigen(JPL(n))
    scatter(1:Nions,λm,label="α="*string(n))
    push!(ratio_ev, λm[Nions]/λm[1])
end
legend()
title("Eigenvalues of \$J^{\\alpha}\$")
xlabel("#");ylabel("λ");
gcf()



v1=[0,4,0,0,0]

v1*v1'

rand(3,0)*rand(0,3)