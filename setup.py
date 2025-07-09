from setuptools import setup, find_packages

setup(
    name="hebbianllm",
    version="0.1.0",
    description="High-performance Hebbian Spiking Neural Network with GPU acceleration",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "jax>=0.4.2",
        "jaxlib>=0.4.2",
        "numpy>=1.24.0",
        "pytest>=7.3.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "hypothesis>=6.75.3",
        ],
        "cuda": [
            "jax[cuda]>=0.4.2",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "plotly>=5.14.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
) 