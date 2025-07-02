from setuptools import setup, find_packages

setup(
    name="gmesh",
    version="0.1",
    description="Differentiable Hybrid Renderer for 3D Gaussians and Meshes",
    author="Peizhi Yan",
    packages=find_packages(),
    install_requires=[
        # "torch",
        # "numpy",
        # "gsplat",
        # "pytorch3d",
    ],
    python_requires=">=3.10",
)