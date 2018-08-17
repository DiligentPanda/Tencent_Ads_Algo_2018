CUDA_PATH=/usr/local/cuda/

cd src/cuda
echo "Compiling reduce kernels by nvcc..."
if [ -f reduce_kernel.cu.o ]
then
 rm reduce_kernel.cu.o
fi
${CUDA_PATH}/bin/nvcc -c -o reduce_kernel.cu.o reduce_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../..
python build.py
