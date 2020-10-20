rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_ENABLE_GPU=OFF ..
cp ../backend_common.cc ./_deps/repo-backend-src/src
make install
cd ..
cp ./build/install/backends/tnn/libtriton_tnn.so ./my_libs/