VERSION >= v"0.4.0-dev+6521" && __precompile__()

module ShapeModels

using MultivariateStats, HDF5, FunctionalDataUtils, DictFiles, Compat

export PCAShapeModel, shape, coeffs, clamp, meanshape, modeshapes, nmodes, vec, modesstd
export axisij, plotshape, plotshapes
export maxcoeffvec, mincoeffvec

immutable PCAShapeModelCoeffs
	modes
	rot
	scale
	translation
    ndims
end

type PCAShapeModel
	aligned
	pca
    ndims
    nlandmarks
    center
    maxtranslation
    buf::Array{Float64,2}
end

modesstd(a::PCAShapeModel) = sqrt(principalvars(a.pca))

function PCAShapeModelCoeffs(a::PCAShapeModel, x)
    if a.ndims == 2
        PCAShapeModelCoeffs(x[1:end-4], x[end-3:end-3], x[end-2], x[end-1:end], a.ndims)
    else
        PCAShapeModelCoeffs(x[1:end-6], x[end-5:end-4], x[end-3], x[end-2:end], a.ndims)
    end
end
import Base.vec
vec(a::PCAShapeModelCoeffs) = [a.modes; a.rot; a.scale; a.translation]


include("plotfunctions.jl")

function PCAShapeModel{T<:Real}(landmarks::Array{T,3}; percentage = 0.98, center = zeros(size(landmarks,1)), 
    maxtranslation = realmax(Float32)*ones(size(landmarks,1)), optcoeffs = false)
    ndims, nlandmarks, nshapes = size(landmarks)

    aligned = copy(landmarks)
    scales = zeros(nshapes)
    aligned = @p map aligned x->x.-mean_(x)
    scales = @p map aligned x->@p distance zeros(sizem(aligned),1) x | mean
    relativescales = scales ./ mean(scales)

    rotmatrices = Array(Any, nshapes)
    rotmatrices[1] = eye(ndims)
    aligned[:,:,1] = aligned[:,:,1] ./= relativescales[1]
    for i = 2:nshapes
        aligned[:,:,i] ./= relativescales[i]
        # https://en.wikipedia.org/wiki/Kabsch_algorithm
        U,E,V = svd(aligned[:,:,1] * aligned[:,:,i]')
        Eprime = eye(length(E))
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
        invrot = @p map rotmatrices inv
        angles = @p map invrot x->asin(x[2,1])
        modes  = @p transform pca aligned | unstack
        translations = @p map landmarks mean_ | map minus center | unstack
        coeffs = @p zip modes angles relativescales-1 translations| collect | map x->PCAShapeModelCoeffs(x[1], x[2], x[3], x[4], ndims)
        model, coeffs
    else
        model
    end
end                                  

import Base.reshape
col(a) = reshape(a, length(a),1)
row(a) = reshape(a, 1, length(a))
reshape(a::PCAShapeModel, b) = reshape(b, a.ndims, a.nlandmarks)
meanshape(a::PCAShapeModel) = shape(a, zeros(nmodes(a)))

maxcoeffvec(a::PCAShapeModel) = vcat(
    2.5*modesstd(a), 
    a.ndims == 2 ? 0.3 : [0.3, 0.3, 0.3],
    0.2,
    a.maxtranslation)
mincoeffvec(a::PCAShapeModel) = -maxcoeffvec(a)

nmodes(a::PCAShapeModel) = outdim(a.pca) + (a.ndims==2 ? 4 : 6)


function modeshapes(a::PCAShapeModel, ind, at = linspace(mincoeffvec(a)[ind], maxcoeffvec(a)[ind],10))
    assert(ind>0 && ind <= nmodes(a))
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
    if sizem(coeffs) != nmodes(a) 
        error("Cant create shape from coeffs, dimension mismatch. size(coeffs) == $(size(coeffs)), should be $(nmodes(a))")
    end
    shape(a, PCAShapeModelCoeffs(a, coeffs))
end
shape(a::PCAShapeModel, coeffs::PCAShapeModelCoeffs) = shape!(a.buf, a, coeffs)
function shape!(buf, a::PCAShapeModel, coeffs::PCAShapeModelCoeffs)
    r = reconstruct(a.pca, coeffs.modes)
    r = reshape(a, r)
    buf[:] = rotmatrix(coeffs)*r .+ coeffs.translation .+ a.center
end

if VERSION.minor == 4
    call(a::PCAShapeModel, coeffs) = shape(a, coeffs)
end

function coeffs{T<:Real}(a::PCAShapeModel, coords::Array{T,2})
    # TODO
end

import Base.clamp
clamp(a::PCAShapeModel, coeffs::Vector) = clamp(a, col(coeffs))
function clamp{T<:Real}(a::PCAShapeModel, coeffs::Array{T,2})
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


end # module
