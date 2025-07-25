import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ext_modules = [
    Extension(
        "module_2_3_1",
        ["module_2_3_1.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/home/aparimit/local/include",  # ← Added this (missing)
            "/home/aparimit/apache-arrow-20.0.0/cpp/src",
            "/home/aparimit/apache-arrow-20.0.0/cpp/build/src",
            "/home/aparimit/h3/build/src/h3lib/include",
            "/home/aparimit/anaconda3/include/python3.11",
            "/home/aparimit/anaconda3/lib/python3.11/site-packages/pybind11/include",
        ],
        library_dirs=[
            "/home/aparimit/local/lib",  # ← Added this (missing)
            "/home/aparimit/apache-arrow-20.0.0/cpp/build/release",
            "/home/aparimit/h3/build/lib",
        ],
        libraries=[
            "arrow",
            "parquet",
            "h3",
            "stdc++fs",
        ],  # No "snappy" to avoid static lib conflict
        language="c++",
        extra_compile_args=[
            "-O3",
            "-std=c++17",
        ],  # No "-fopenmp" to avoid threading conflicts
        extra_link_args=[
            "/home/aparimit/anaconda3/lib/libpython3.11.so",
            "/home/aparimit/anaconda3/lib/libsnappy.so.1",  # Explicit shared snappy
            "-pthread",
        ],
    ),
]

setup(
    name="module_2_3_1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
