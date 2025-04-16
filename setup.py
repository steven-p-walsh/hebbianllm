from setuptools import setup, find_packages

setup(
    name="hebbianllm",
    version="0.1.0",
    description="Biologically-inspired Hebbian Spiking Neural Network for language modeling",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "jax>=0.4.2",
        "jaxlib>=0.4.2",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pytest>=7.3.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "hypothesis>=6.75.3",
        ],
        "cuda": [
            "cupy>=12.0.0",
        ],
        "visualization": [
            "plotly>=5.14.0",
            "dash>=2.9.0",
            "networkx>=3.1",
        ],
    },
) 