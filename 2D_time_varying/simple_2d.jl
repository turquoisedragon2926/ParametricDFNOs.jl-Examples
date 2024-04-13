using MPI
using CUDA
using Zygote
using ParametricDFNOs.DFNO_2D

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

# Julia requires you to manually assign the gpus, modify to your case.
CUDA.device!(rank % 4)
partition = [1, size]

@assert MPI.Comm_size(comm) == prod(partition)
modelConfig = DFNO_2D.ModelConfig(nx=20, ny=20, nt=30, mx=4, my=4, mt=4, nblocks=4, partition=partition, dtype=Float32)

model = DFNO_2D.Model(modelConfig)
θ = DFNO_2D.initModel(model)

input_size = (model.config.nc_in * model.config.nx * model.config.ny * model.config.nt) ÷ prod(partition)
output_size = (input_size * model.config.nc_out ÷ model.config.nc_in) ÷ prod(partition)

x_sample = rand(modelConfig.dtype, input_size, 1)
y_sample = cu(rand(modelConfig.dtype, output_size, 1))

@time y = DFNO_2D.forward(model, θ, x_sample)
@time y = DFNO_2D.forward(model, θ, x_sample)
@time y = DFNO_2D.forward(model, θ, x_sample)

function loss_helper(params)
    global loss = UTILS.dist_loss(DFNO_2D.forward(model, params, x_sample), y_sample)
    return loss
end

rank == 0 && println("STARTED GRADIENT SCALING")

@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]
@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]
@time grads_time = @elapsed gradient(params -> loss_helper(params), θ)[1]

MPI.Finalize()
