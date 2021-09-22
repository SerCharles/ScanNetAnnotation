export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64


export DFLAGS="-L/usr/local/cuda/lib64"

# build source
g++ -std=c++11 -c main.cpp $CFLAGS -O2 -o main.o -fPIC
nvcc -std=c++11 -c annotate.cu -D_FORCE_INLINES -O2 -o annotate.o --compiler-options '-fPIC'

# test buffer
g++ -std=c++11 main.o annotate.o $DFLAGS -O2 -o Annotator.so -shared -fPIC -lcudart