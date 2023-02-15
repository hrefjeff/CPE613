# Ubuntu

jeff

1234

# Local Install of CUDA

https://docs.nvidia.com/cuda/wsl-user-guide/index.html This documentation got me up and running in no time

https://www.youtube.com/watch?v=1HGBk78BqR4 Helpful article to get debugger for WSL

```sh
LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
```

## To Uninstall

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.7/bin

## Include Paths

```sh
CUDAPATH=/usr/local/cuda-12.0
SAMPLESPATH=/home/jeff/code/cuda-samples/Common
```

If "cuda_helper.h" can't be found, update the code with the SAMPLESPATH. If we don't have the samples folder, grab it from github and download it to the cuda directory

```sh
git clone https://github.com/nvidia/cuda-samples
```


# Windows

Windows Visual Studio 2022 can run the CUDA debugger on my laptop. WSL cannot and doesn't have support for it.
Using the cuda examples I was able to stop the program on kernel launch and switch focus' to view variable info in a specific thread
