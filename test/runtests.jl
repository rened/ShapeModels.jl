println("Running runtests.jl ...")
push!(LOAD_PATH, "..")

using FunctionalData
using HDF5
using MultivariateStats
using ShapeModels
using Statistics
using Test

macro shouldtestset(a,b) length(ARGS) < 1 || ARGS[1] == a ?  :(@testset $a $b) : nothing end
macro shouldtestset2(a,b) length(ARGS) < 2 || ARGS[2] == a ?  :(@testset $a $b) : nothing end

landmarks = ShapeModels.examplelandmarks()

@shouldtestset "basic" begin
	a = PCAShapeModel(landmarks)
	@test size(projection(a.pca)) == (256,8)
	@test size(principalvars(a.pca)) == (8,)
	@test indim(a.pca) == 256
	@test outdim(a.pca) == 8
end

@shouldtestset "utils" begin
	a = PCAShapeModel(landmarks)
	m = meanshape(a)
	@test size(m) == (2,128)
	@test maximum(abs.(mean(m, dims = 2)))  < 1e-6
end

@shouldtestset "shape" begin
	a = PCAShapeModel(landmarks)

    @test size(shape(a, zeros(nmodes(a)))) == (2,128)
end

@shouldtestset "modes" begin
	a = PCAShapeModel(landmarks)
	@test size(modeshapes(a, 1)) == (2,128,10)
end




println("  ... finished runtests.jl")

