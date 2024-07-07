# ./amalgamation/Heady_amalgate -s src -e "config.h compile.sh rnn_train.py write_weights.c rnnoise_data.c vec_avx.h"  -o "header_only/rnnoise_amalgamated.c" -d RNNOISE_ALMAGAMATED

import pathlib
import subprocess
import os

SOURCE_PATH = pathlib.Path(os.getcwd(), "amalgamation", "Heady")
BUILD_PATH = pathlib.Path(os.getcwd(), "amalgamation", "Heady", "build")


def build_if_necessary():
    cmake_configure = [
        "cmake",
        "-G",
        "Unix Makefiles",
        f"-S={SOURCE_PATH}",
        f"-B={BUILD_PATH}",
    ]
    subprocess.run(
        cmake_configure,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    cmake_build = [
        "cmake",
        "--build",
        f"{pathlib.Path(os.getcwd(),'amalgamation','Heady','build')}",
    ]

    subprocess.run(
        cmake_build,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def run_shell_command():
    command = [
        f"{pathlib.Path(BUILD_PATH,'Heady')}",
        "-s",
        "src",
        "-e",
        "config.h compile.sh rnn_train.py write_weights.c rnnoise_data.c vec_avx.h",
        "-o",
        "header_only/rnnoise_amalgamated.c",
        "-d",
        "RNNOISE_ALMAGAMATED",
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("Command executed successfully.")
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred while executing the command.")
        print("Error message:\n", e.stderr)


if __name__ == "__main__":
    build_if_necessary()
    run_shell_command()
