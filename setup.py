from setuptools import setup, find_packages

setup(
    name="edgeformer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "datasets>=2.12.0",
        "optimum>=1.8.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    author="Oscar Nunez",
    author_email="art.by.oscar.n@gmail.com",
    description="Optimized transformer model with Multi-Head Latent Attention for edge devices",
    keywords="transformer, nlp, optimization, edge, mla",
    python_requires=">=3.8",
)