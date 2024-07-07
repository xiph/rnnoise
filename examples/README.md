## Building with Conan
```shell
pip install conan
mkdir build

## For debug version of the app
conan install conanfile.txt --build=missing --settings=build_type=Debug
cmake -G"Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=./Debug/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -B build
cmake --build build

## For Release version:
conan install conanfile.txt --build=missing
cmake -G"Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=./Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DRNNOISE_COMPILE_DEMO=ON -B build
cmake --build build
```