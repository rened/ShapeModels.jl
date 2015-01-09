module ShapeModels

export PCAShapeModel, shape, coeffs, clamp, plotshapes

type PCAShapeModel

	aligned
    mean
    eigvec
    eigval
    ndims
    nlandmarks
end

include("utils.jl")

function PCAShapeModel{T<:Real}(landmarks::Array{T,3}, percentage = 0.95)
    ndims, nlandmarks, nshapes = size(landmarks)

    aligned = copy(landmarks)
    scales = zeros(nshapes)
    for i = 1:nshapes
        aligned[:,:,i] .-= mean(aligned[:,:,i],2)
        scales[i] = mean(sqrt(sum(aligned[:,:,i].^2,1)))
    end
    relativescales = scales / mean(scales)

    for i = 2:nshapes
        aligned[:,:,i] ./= relativescales[i]
        # https://en.wikipedia.org/wiki/Kabsch_algorithm
        U,E,V = svd(aligned[:,:,1] * aligned[:,:,i]')
        Eprime = eye(length(E))
        Eprime[end] = sign(det(U*V))
        rotmatrix = V' * Eprime' * U
        #@show rotmatrix
        aligned[:,:,i] = rotmatrix * aligned[:,:,i]
    end
    
	alignedshapes = aligned
    aligned = reshape(aligned, (ndims*nlandmarks, nshapes))
    m = mean(aligned, 2)
    eigvec, eigval = pca(aligned .- m)
    nmodes = findfirst(cumsum(eigval)/sum(eigval) .>= percentage)
    PCAShapeModel(alignedshapes, m, eigvec, eigval, ndims, nlandmarks)
end

import Base.reshape
reshape(a::PCAShapeModel, b) = reshape(b, (s.ndims, a.mean/s.ndims, size(b,1))) 
meanshape(a::PCAShapeModel) = reshape(a, a.mean)
modes(a::PCAShapeModel, ind = 1:size(a.eigvec,2)) = reshape(a, eigvec[:,ind])

# coefficients are 
#   2D: dm, dn, scale, roto
#   3D: dm, dn, do, scale, rotm, rotn, roto
ncoeffs(a::PCAShapeModel) = a.ndims==2 ? 4 : 7

column(a::Vector) = reshape(a, (length(a),1))
rotmartix(a::PCAShapeModel, coeffs::Vector) = rotmatrix(a, column(coeffs))
rotmartix{T<:Real}(a::PCAShapeModel, coeffs::Array{T,2}) = rotmatrix(coeffs[a.ndims==2 ? 3:4 : 4:7]...)
rotmatrix(scaling, alpha) = scaling*[cos(alpha) -sin(alpha); sin(alpha) cos(alpha)]
function rotmatrix(scaling, rotm, rotn, roto) 
    Rm = [1 0 0; 0 cos(rotm) -sin(rotm); 0 sin(rotm) cos(rotm)]
    Rn = [cos(rotn) 0 sin(rotn); 0 1 0; -sin(rotn) 0 cos(rotn)]
    Ro = [cos(roto) -sin(roto) 0; sin(roto) cos(roto) 0; 0 0 1]
    scaling*Rm*Rn*Ro
end

shape(a::PCAShapeModel, coeffs::Vector) = shape(a, column(coeffs))
function shape{T<:Real}(a::PCAShapeModel, coeffs::Array{T,2})
    r = a.eigvec*coeffs[5:end] .+ a.mean
    r = reshape(r, (a.ndims, a.nlandmarks, size(coeffs,2)))
    r = rotmatrix(a, coeffs)*r .+ coeffs[1:a.ndims]
end

function coeffs{T<:Real}(a::PCAShapeModel, coords::Array{T,2})
    # TODO
end

clamp(a::PCAShapeModel, coeffs::Vector) = clamp(a, column(coeffs))
function clamp{T<:Real}(a::PCAShapeModel, coeffs::Array{T,2})
    r = min()
end

end # module
