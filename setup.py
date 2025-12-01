"""Setup script for context_lab package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="context_lab",
    version="1.0.0",
    author="Boris Volovelsky",
    author_email="bvolovelsky@example.com",
    description="LLM Context Window Analysis Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/volo10/context_lab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.1",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "viz": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "context-lab=context_lab.context_lab:main",
        ],
    },
    include_package_data=True,
    package_data={
        "context_lab": ["*.md", "plots/*.png"],
    },
    keywords="llm context-window rag retrieval-augmented-generation nlp ai",
    project_urls={
        "Bug Reports": "https://github.com/volo10/context_lab/issues",
        "Source": "https://github.com/volo10/context_lab",
        "Documentation": "https://github.com/volo10/context_lab/blob/main/README.md",
    },
)

