println("Running runtests.jl ...")
using ShapeModels, MultivariateStats, FactCheck, HDF5
FactCheck.setstyle(:compact)

shouldtest(f, a) = length(ARGS) == 0 || in(a, ARGS) ? facts(f, a) : nothing
shouldtestcontext(f, a) = length(ARGS) < 2 || a == ARGS[2] ? context(f, a) : nothing
 
macro throws_pred(ex) FactCheck.throws_pred(ex) end

landmarks = h5read(joinpath(dirname(@__FILE__()),"../data/2Dlandmarks.hdf5"),"landmarks")

shouldtest("basic") do
	a = PCAShapeModel(landmarks)
	@fact size(projection(a.pca)) => (256,5)
	@fact size(principalvars(a.pca)) => (5,)
	@fact indim(a.pca) => 256
	@fact outdim(a.pca) => 5
end

shouldtest("utils") do
	a = PCAShapeModel(landmarks)
	m = meanshape(a)
	@fact size(m) => (2,128)
	@fact maximum(abs(mean(m,2))) => less_than(1e-6)
end

shouldtest("shape") do
	a = PCAShapeModel(landmarks)

    @fact size(shape(a, zeros(nmodes(a)))) => (2,128)
end

shouldtest("modes") do
	a = PCAShapeModel(landmarks)
	@fact size(modeshapes(a, 1)) => (2,128,10)
end




println("  ... finished runtests.jl")

