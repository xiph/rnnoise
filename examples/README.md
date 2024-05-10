## Building with Conan
```shell
pip install conan
mkdir build

## For debug version of the app
conan install conanfile.txt --build=missing --settings=build_type=Debug
cd build
cmake -G"Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=./Debug/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug ..


## For Release version:
conan install conanfile.txt --build=missing
cd build
cmake -G"Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=./Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release ..


cmake --build .
```