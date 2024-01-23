nvcc -g -G -std=c++17 -lcurand ./src/main.cu -I./src/ -I./src/vendor/ -o Raytracer_Cuda.ds && ./Raytracer_Cuda.ds
