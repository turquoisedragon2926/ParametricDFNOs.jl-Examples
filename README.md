# ParametricDFNOs.jl-Examples

This is a copy of the [`examples`](https://github.com/slimgroup/ParametricDFNOs.jl/tree/master/examples) directory of [`ParametricDFNOs.jl`](https://slimgroup.github.io/ParametricDFNOs.jl) to facilitate frictionless reproducibility.

To get started:

```shell
git clone https://github.com/turquoisedragon2926/ParametricDFNOs.jl-Examples.git
cd ParametricDFNOs.jl-Examples
```

Initialize the dependencies by running the following in a julia REPL:

```julia
julia> ]
(v1.9) activate .
(ParametricDFNOs.jl-Examples) instantiate
```

## Distributed 2D Time Varying FNO:

> [!WARNING]  
> Make sure to have a proper MPI distribution loaded.

### Simple Forward and Gradient Pass
If you have [`mpiexecjl`](https://juliaparallel.org/MPI.jl/stable/usage/#Installation) set up, you can do:

```shell
mpiexecjl --project=./ -n NTASKS julia 2D_time_varying/simple_2d.jl
```

OR if you have a HPC cluster with [`slurm`](https://slurm.schedmd.com/documentation.html) set up, you can do:

```shell
salloc --gpus=NTASKS --time=01:00:00 --ntasks=NTASKS --gpus-per-task=1 --gpu-bind=none
```

> [!WARNING]  
> Your `salloc` might look different based on your HPC cluster.

Now run any of the distributed examples:

### Simple Forward and Gradient Pass

```shell
srun julia --project=./ 2D_time_varying/simple_2d.jl
```

### Training to predict CO2 saturation (dataset provided)

```shell
srun julia --project=./ 2D_time_varying/training_2d.jl
```

## Distributed 3D Time Varying FNO:

### Simple Forward and Gradient Pass

```shell
mpiexecjl --project=./ -n NTASKS julia 2D_time_varying/simple_2d.jl
```

OR

```shell
srun julia --project=./ 2D_time_varying/simple_2d.jl
```

### Training to predict CO2 saturation (example)

```
3D_time_varying/custom
├── data.jl
├── train.jl
└── train.sh
```

We do not provide the dataset, but this is an implementation example for handling complex data storage scheme. See [this]() for more documentation.
