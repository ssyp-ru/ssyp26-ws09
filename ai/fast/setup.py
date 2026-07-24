from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

ext_modules = [
    Extension(
        'oar_core',
        ['src/oar.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17']
    ),
    Extension(
        'bsc_core',
        ['src/bsc.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17']
    )
]


setup(
    name='crabos_core',
    version='0.1.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    setup_requires=[
        "numpy<2.0.0",
        "pybind11"
    ],
    install_requires=[
        "numpy<2.0.0",
        "pybind11"
    ]
)
