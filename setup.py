from setuptools import setup, find_packages

setup(
    name="gmesh",
    version="1.0",
    description="Differentiable Hybrid Renderer for 3D Gaussians and Meshes",
    author="Peizhi Yan",
    packages=find_packages(),
    package_data={"gmesh": ["renderer/csrc/*.cpp", "renderer/csrc/*.cu"]},
    install_requires=[
        # "torch",
        # "numpy",
        # "gsplat",
    ],
    python_requires=">=3.10",
)