cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_ENABLE_GPU=OFF ..
make install
cd ..
cp ./build/install/backends/tnn/libtriton_tnn.so ./my_libs/