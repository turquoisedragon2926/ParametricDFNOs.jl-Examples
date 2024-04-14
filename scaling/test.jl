using ParametricDFNOs.DFNO_3D
using ParametricDFNOs.UTILS
using MPI
using Zygote
using DrWatson
using CUDA

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
pe_count = MPI.Comm_size(comm)

println("Rank: ", rank)
MPI.Barrier(comm)

global gpu_flag = parse(Bool, get(ENV, "DFNO_3D_GPU", "0"))
DFNO_3D.set_gpu_flag(gpu_flag)

println("Flag: ", DFNO_3D.gpu_flag)
DFNO_3D.gpu_flag && (CUDA.device!(rank % 4))

println(CUDA.device())

MPI.Barrier(comm)

MPI.Finalize()
