g++ -g -std=c++17 ./src/main.cpp -lhiredis -I./src/ -I./src/vendor/ -I/usr/local/include/ -lpqxx -lpq  -ltbb -o Raytracer_Redis.ds && ./Raytracer_Redis.ds
