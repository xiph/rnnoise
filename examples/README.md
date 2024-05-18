## Building with Conan
```shell
pip install conan
mkdir build

## For debug version of the app
conan install conanfile.txt --build=missing --settings=build_type=Debug
cmake -G"Unix Makefiles" -B=build --preset=conan-debug .


## For Release version:
conan install conanfile.txt --build=missing
cd build
cmake -G"Unix Makefiles" -B=build --preset=conan-release .


cmake --build .
```