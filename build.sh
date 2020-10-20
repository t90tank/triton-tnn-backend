rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_ENABLE_GPU=OFF ..
make install