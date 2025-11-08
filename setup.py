from setuptools import setup, find_packages

setup(
    name="pci_torch",
    version="0.1.0",
    description="GPU-accelerated PCI forward simulation using PyTorch",
    author="PCI Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "h5py>=3.8.0",
    ],
    python_requires=">=3.8",
)



