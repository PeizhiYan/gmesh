from setuptools import setup, find_packages

setup(
    name="gmesh",
    version="1.0",
    description="Differentiable Hybrid Renderer for 3D Gaussians and Meshes",
    author="Peizhi Yan",
    packages=find_packages(),
    package_data={"gmesh": [
        "renderer/csrc/*.cpp", "renderer/csrc/*.cu",
        # adapted from gsplat (Apache-2.0)
        "renderer/csrc/gsplat/*",
        # GLM math library (MIT)
        "renderer/csrc/third_party/glm/copying.txt",
        "renderer/csrc/third_party/glm/glm/**/*",
    ]},
    install_requires=[
        # "torch",
        # "numpy",
    ],
    python_requires=">=3.10",
)