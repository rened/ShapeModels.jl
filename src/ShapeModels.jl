module ShapeModels

using HDF5
using LinearAlgebra
using MultivariateStats
using Pkg
using Plots
using Statistics

export PCAShapeModel, shape, coeffs, clamp, meanshape, modeshapes, nmodes, vec, modesstd
export axisij, plotshape, plotshapes
export maxcoeffvec, mincoeffvec

struct PCAShapeModelCoeffs
	modes
	rot
	scale
	translation
    ndims
end

struct PCAShapeModel
	aligned
	pca
    ndims
    nlandmarks
    center
    maxtranslation
    buf::Array{Float64,2}
end

modesstd(a::PCAShapeModel) = sqrt.(principalvars(a.pca))

function PCAShapeModelCoeffs(a::PCAShapeModel, x)
    if a.ndims == 2
        PCAShapeModelCoeffs(x[1:end-4], x[end-3:end-3], x[end-2], x[end-1:end], a.ndims)
    else
        PCAShapeModelCoeffs(x[1:end-6], x[end-5:end-4], x[end-3], x[end-2:end], a.ndims)
    end
end
import Base.vec
vec(a::PCAShapeModelCoeffs) = [a.modes; a.rot; a.scale; a.translation]

eye(a) = diagm(0 => ones(a))
function PCAShapeModel(landmarks::Array{T,3};
                       percentage = 0.98,
                       center = zeros(size(landmarks,1)), 
                       maxtranslation = typemax(Float32)*ones(size(landmarks,1)),
                       optcoeffs = false) where T<:Real

    ndims, nlandmarks, nshapes = size(landmarks)

    aligned = copy(landmarks)
    scales = zeros(nshapes)
    aligned = aligned .- mean(aligned, dims = 2)
    scales = mean(mapslices(norm, aligned, dims=1), dims=(1,2))[1,1,:]
    relativescales = scales ./ mean(scales)

    rotmatrices = Array{Any}(undef, nshapes)
    rotmatrices[1] = I(ndims)
    aligned[:,:,1] = aligned[:,:,1] ./= relativescales[1]
    for i = 2:nshapes
        aligned[:,:,i] ./= relativescales[i]
        # https://en.wikipedia.org/wiki/Kabsch_algorithm
        U,E,V = svd(aligned[:,:,1] * aligned[:,:,i]')
        Eprime = I(length(E))
        Eprime[end] = sign(det(U*V))
        rotmatrices[i] = V' * Eprime' * U
        aligned[:,:,i] = rotmatrices[i] * aligned[:,:,i]
    end
    
	alignedshapes = aligned
    aligned = reshape(aligned, (ndims*nlandmarks, nshapes))
	pca = fit(PCA, aligned, pratio = percentage)

    model = PCAShapeModel(alignedshapes, pca, ndims, nlandmarks, center, maxtranslation, zeros(ndims, nlandmarks))

    ## compute optimal coeffs for training data
    if optcoeffs
        invrot = inv.(rotmatrices)
        angles = map(x->asin(x[2,1]), invrot)
        modes  = unstack(transform(pca, aligned))
        translations = unstack(mean(landmarks, dims = 2) .- center)
        input = collect(zip(modes, angles, relativescales.-1, translations))
        coeffs = map(x->PCAShapeModelCoeffs(x[1], x[2], x[3], x[4], ndims), input)
        model, coeffs
    else
        model
    end
end                                  

import Base.reshape
col(a) = reshape(a, length(a),1)
row(a) = reshape(a, 1, length(a))
unstack(a) = Any[at(a,i) for i in 1:len(a)]
reshape(a::PCAShapeModel, b) = reshape(b, a.ndims, a.nlandmarks)
meanshape(a::PCAShapeModel) = shape(a, zeros(nmodes(a)))

maxcoeffvec(a::PCAShapeModel) = vcat(
    2.5*modesstd(a), 
    a.ndims == 2 ? 0.3 : [0.3, 0.3, 0.3],
    0.2,
    a.maxtranslation)
mincoeffvec(a::PCAShapeModel) = -maxcoeffvec(a)

nmodes(a::PCAShapeModel) = outdim(a.pca) + (a.ndims==2 ? 4 : 6)


function modeshapes(a::PCAShapeModel, ind, at = range(mincoeffvec(a)[ind], stop = maxcoeffvec(a)[ind], length = 10))
    @assert ind>0 && ind <= nmodes(a)
	r = zeros(a.ndims, a.nlandmarks, length(at))
	for i = 1:length(at)
    	v = zeros(nmodes(a))
		v[ind] = at[i]
		r[:,:,i] = shape(a, v)
	end
	r
end

rotmatrix(coeffs::PCAShapeModelCoeffs) = rotmatrix(coeffs.scale, coeffs.rot...)
rotmatrix(scale, alpha) = (scale+1)*[cos(alpha) -sin(alpha); sin(alpha) cos(alpha)]
function rotmatrix(scale, rotm, rotn, roto) 
    Rm = [1 0 0; 0 cos(rotm) -sin(rotm); 0 sin(rotm) cos(rotm)]
    Rn = [cos(rotn) 0 sin(rotn); 0 1 0; -sin(rotn) 0 cos(rotn)]
    Ro = [cos(roto) -sin(roto) 0; sin(roto) cos(roto) 0; 0 0 1]
    (scale+1)*Rm*Rn*Ro
end

function shape(a::PCAShapeModel, coeffs)
    if size(coeffs) != (nmodes(a),) 
        error("Cant create shape from coeffs, dimension mismatch. size(coeffs) == $(size(coeffs)), should be ($(nmodes(a)),)")
    end
    shape(a, PCAShapeModelCoeffs(a, coeffs))
end
shape(a::PCAShapeModel, coeffs::PCAShapeModelCoeffs) = shape!(a.buf, a, coeffs)
function shape!(buf, a::PCAShapeModel, coeffs::PCAShapeModelCoeffs)
    r = reconstruct(a.pca, coeffs.modes)
    r = reshape(a, r)
    buf[:] = rotmatrix(coeffs)*r .+ coeffs.translation .+ a.center
end

(a::PCAShapeModel)(coeffs) = shape(a, coeffs)

function coeffs(a::PCAShapeModel, coords::Array{T,2}) where T<:Real
    # TODO
end

import Base.clamp
clamp(a::PCAShapeModel, coeffs::Vector) = clamp(a, col(coeffs))
function clamp(a::PCAShapeModel, coeffs::Array{T,2}) where T<:Real
    # TODO
    error("write me")
end

examplelandmarks() = examplelandmarks(:hands2d)
function examplelandmarks(a::Symbol)
    if a==:hands2d
        return h5read(joinpath(dirname(@__FILE__()),"../data/2Dlandmarks.hdf5"),"landmarks")
    elseif a==:lungs
        return dictread(joinpath(dirname(@__FILE__()),"../data/lungs.dictfile"))[:landmarks]
    else
        error("unknown dataset '$a'")
    end
end

function exampleimages(a::Symbol)
    if a == :hands2d
        shapes = examplelandmarks(a)
        return [inpolygon(at(shapes,i).-mean(at(shapes,i)).+[200 250]',zeros(500,400)) for i in 1:14]
    else
        error("unknown dataset '$a'")
    end
end


function axisij() 
    a = axis() 
    axis((a[1],a[2],a[4],a[3]))
end

function plotshape(a, args...; kargs...)
    if size(a,1)==2
        plot(vec(a[2,:]), vec(a[1,:]), args...; kargs...)
        # axisij()
        # xlabel("N")
        # ylabel("M")
    elseif size(a,1)==3
        plot3d(vec(a[1,:]), vec(a[2,:]), vec(a[3,:]), args...; kargs...)
        # xlabel("M")
        # ylabel("N")
        # zlabel("O")
    else
        error("Can't plot data of size $(size(a))")
    end
    # axis("equal")
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
            plotshape(at(a,round(Int,datai)))
        end
    end
end

end # module
