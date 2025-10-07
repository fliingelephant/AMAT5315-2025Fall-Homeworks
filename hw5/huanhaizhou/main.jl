# Homework 5

include("../download_mnist.jl")
FOLDER = "hw5"

## Q1
train_images, train_labels = download_mnist(:train)
test_images, test_labels = download_mnist(:test)

### Vectorize the images (flatten each 28×28 image into a 784-dimensional vector).
### Construct a data matrix where each column (or row) represents a flattened image.
train_images_vectorized = reshape(train_images, prod(size(train_images)[1:2]), :)

### Apply SVD to the data matrix.
using LinearAlgebra
function truncated_svd(M::AbstractMatrix; atol::Real=0.0, maxdim::Int=typemax(Int))
    @assert atol >= zero(atol)
    @assert maxdim > 0
    
    res = LinearAlgebra.svd(M)
    r = min(searchsortedfirst(res.S, atol; rev=true) - 1, maxdim, length(res.S))

    return res.U[:, 1:r], res.S[1:r], res.Vt[1:r, :]
end

### Compress the dataset by retaining only the top k singular values
ks = [10, 50, 100, 200]
train_images_compressed = (k->truncated_svd(train_images_vectorized; maxdim=k)).(ks)

### Reconstruct the images from the compressed representation.
train_images_reconstructed = [
    U * Diagonal(S) * V
    for (U, S, V) in train_images_compressed
]
train_images_reconstructed = (x->reshape(x, 28, 28, :)).(train_images_reconstructed)

### Visualize and compare the original vs. reconstructed images.
using Makie, CairoMakie
imggrid(A::AbstractArray{<:Any,4}) =
    reshape(permutedims(A, (1,3,2,4)), size(A,1)*size(A,3), size(A,2)*size(A,4))
nrows, ncols = 8, 8
fig = Makie.Figure(resolution=(100*ncols, 100*nrows))
idx = rand(1:size(train_images, 3), nrows * ncols) # random indices of digits

#### Original images
ax = Makie.Axis(fig[1,1], yreversed=true, title="Original images")
Makie.image!(ax, imggrid(reshape(train_images[:,:,idx], 28, 28, ncols, nrows)), colorrange=(0, 1))
Makie.hidedecorations!(ax)
Makie.hidespines!(ax)

#### Reconstructed images
positions = CartesianIndices((1:2, 1:(length(ks) ÷ 2 + 1))) .+ CartesianIndex(1,0)
for (k, M, pos) in zip(ks, train_images_reconstructed, positions[eachindex(ks)])
    ax = Makie.Axis(fig[Tuple(pos)...], yreversed=true, title="k=$k")
    Makie.image!(ax, imggrid(reshape(M[:,:,idx], 28, 28, ncols, nrows)), colorrange=(0, 1))
    Makie.hidedecorations!(ax)
    Makie.hidespines!(ax)
end
fig
save(joinpath(FOLDER, "huanhaizhou", "svd_reconstruction.png"), fig)

#### Report the compression ratio and reconstruction error (e.g., mean squared error or Frobenius norm).
compression_ratios, reconstruction_errors = [], []
for (k, M) in zip(ks, train_images_reconstructed)
    push!(compression_ratios, (size(train_images, 1) + k + size(train_images, 2)) * k / length(train_images))
    push!(reconstruction_errors, norm(train_images .- M))
    @info "k=$k, Compression ratio: $(compression_ratios[end]), Reconstruction error: $(reconstruction_errors[end]) in Frobenius norm"
end
fig = Makie.Figure(resolution=(600, 800))
ax = Makie.Axis(fig[1,1], xlabel="#Retained singular values", ylabel="Compression ratio", xticks=ks)
Makie.plot!(ax, ks, compression_ratios, label="SVD")
ax2 = Makie.Axis(fig[2,1], xlabel="#Retained singular values", ylabel="Reconstruction error", xticks=ks)
Makie.plot!(ax2, ks, reconstruction_errors, label="SVD")
fig
save(joinpath(FOLDER, "huanhaizhou", "svd_compression_ratio_and_reconstruction_error.png"), fig)

#### Discussion
# As the number of retained singular values increases, the compression ratio increases (in quadratic law) and the reconstruction error decreases.
# Better compression ratio, worse image quality.


## Q2 Part A
using FFTW, Images, ImageEdgeDetection

function truncated_fft(A::AbstractArray; atol::Real=0.0, maxdim::Int=typemax(Int))
    @assert atol >= zero(atol)
    @assert maxdim > 0

    fft_A = fft(A)
    norm_fft_A = sort(reshape(norm.(fft_A), :))

    r = searchsortedfirst(norm_fft_A, atol)
    r > length(norm_fft_A) && return fft_A .* falses(size(A)), falses(size(A))
    r = max(r, length(norm_fft_A) - maxdim + 1)

    filter = x::Real-> norm_fft_A[r]<=x
    mask = filter.(norm.(fft_A))

    return fft_A .* mask, mask
end

function Fourier_compression(img::AbstractArray{<:Real, N}; ratio::Float64=1e-2, mask::BitArray{N}=trues(size(img))) where N
    fft_img, _ = truncated_fft(img; maxdim=floor(Int, ratio * length(img)))
    return real.(ifft(fft_img .* mask))
end

function edge_guided_Fourier_compression(img::AbstractArray{<:Real}; algorithm::ImageEdgeDetection.AbstractEdgeDetectionAlgorithm, ratio::Float64=1e-2)
    _, mask = truncated_fft(detect_edges(img, algorithm); maxdim=floor(Int, ratio * length(img)))
    return Fourier_compression(img; ratio=1.0, mask=mask)
end

function svd_compression(img::AbstractMatrix{<:Real}; ratio::Float64=1e-2)
    U, S, V = truncated_svd(img; maxdim=floor(Int, ratio * length(img)))
    return U * Diagonal(S) * V
end

function edge_guided_svd_compression(img::AbstractMatrix{<:Real}; algorithm::ImageEdgeDetection.AbstractEdgeDetectionAlgorithm, atol::Float64=1e-4)
    U, _, V = truncated_svd(detect_edges(img, algorithm); atol=atol)
    return U * U' * img * V' * V
end


img = load(joinpath(FOLDER, "cat.png"))
pics = channelview(img)
pics = [pics[i, :, :] for i in axes(pics, 1)]


fourier_compressed = stack(Fourier_compression.(pics; ratio=1e-2); dims=1)
img = colorview(RGB, fourier_compressed)

svd_compressed = stack(svd_compression.(pics; ratio=1e-3); dims=1)
img = colorview(RGB, svd_compressed)

edge_guided_fourier_compressed = stack(edge_guided_Fourier_compression.(pics; algorithm=ImageEdgeDetection.Canny(spatial_scale=1.2, high=ImageEdgeDetection.Percentile(70), low=ImageEdgeDetection.Percentile(30)), ratio=1e-1); dims=1)
img = colorview(RGB, edge_guided_fourier_compressed)

edge_guided_svd_compressed = stack(edge_guided_svd_compression.(pics; algorithm=ImageEdgeDetection.Canny(spatial_scale=1.2), atol=1e-1); dims=1)
img = colorview(RGB, edge_guided_svd_compressed)
