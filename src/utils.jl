isinstalled(a) = isa(Pkg.installed(a), VersionNumber)
if isinstalled("PyPlot")
	using PyPlot


function plotshape(a)
    if size(a,1)==2
        plot(vec(a[2,:]), vec(a[1,:]), marker=".", linestyle="")
        xlabel("N")
        ylabel("M")
    elseif size(a,1)==3
        plot3D(vec(a[1,:]), vec(a[2,:]), vec(a[3,:]), marker=".", linestyle="")
        xlabel("M")
        ylabel("N")
        zlabel("O")
    else
        error("Can't plot data of size $(size(a))")
    end
    axis("equal")
end

function plotshapes(a)
	if size(a,1)==3
		for i = 1:last(size(a))
			plotshape(a[:,:,i])
			hold(true)
		end
		hold(false)
	else
		gridplot(a)
	end
end

at(a,i) = slicedim(a,ndims(a),i)

function gridplot(a)
    N = last(size(a))
    sm = floor(sqrt(N))
    sn = ceil(N/sm)
    for m = 1:sm, n = 1:sn
        datai = (n-1)*sm+m
        ploti = (m-1)*sn+n
        if datai <= N
            subplot(sm,sn,ploti)
            plotshape(at(a,datai))
        end
    end
end

end

function pca(X, A = min(size(X)...), normmean = true)
    # [Z,U,l] = pca(X, A)
    # pca analysis
    # X   : Centred dataset (I x J)
    # A   : Number of principal components
    #
    # Z   : z-scores (I x A)
    # U   : Eigenvectors (J x A)
    # l   : Eigenvalues (I)

    I, J = size(X)
    
	me = mean(X,2)
    if normmean
        X .-= me
    else
		me *= 0
    end

    # The 'economy size' svd demands that
    # X has more rows than columns
    if I > J
        u,s,v = svd(X)
        S = diagm(s)
        Z = u*S
        Z = Z[:,1:A]
        U = v[:,1:A]
    else
        u,s,v = svd(X')
        S = diagm(s)
        Z = v*S'
        Z = Z[:,1:A]
        U = u[:,1:A]
    end

    l = s.^2/(I-1)

    U, l, me, Z
end

