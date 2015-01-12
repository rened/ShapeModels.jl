isinstalled(a) = isa(Pkg.installed(a), VersionNumber)
if isinstalled("PyPlot")
	using PyPlot

    function axisij() 
        a = axis() 
        axis((a[1],a[2],a[4],a[3]))
    end

	function plotshape(a, args...; kargs...)
		if size(a,1)==2
			plot(vec(a[2,:]), vec(a[1,:]), args...; kargs...)
            axisij()
			xlabel("N")
			ylabel("M")
		elseif size(a,1)==3
			plot3D(vec(a[1,:]), vec(a[2,:]), vec(a[3,:]), args...; kargs...)
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

