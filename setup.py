from setuptools import setup

version="1.0.0"

with open("README.md", "r") as fd:
    long_description = fd.read()

setup(
    name="mp_shared_memory",
    version=version,
    description="Multiprocessing shared memory",
    long_description=long_description,
    author="Kwanghun Chung Lab",
    packages=["blockfs"],
    url="https://github.com/chunglabmit/mp_shared_memory",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        'Programming Language :: Python :: 3.5'
    ],
    install_requires=[
        "numpy"
    ]
)