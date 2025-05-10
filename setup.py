from setuptools import setup, find_packages

setup(
    name="moroccan_income_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "sweetviz>=2.1.0",
        "joblib>=1.2.0",
        "jupyter>=1.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning project for predicting annual income of Moroccans",
    long_description=open("README.md").read() if open("README.md").read() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/moroccan_income_prediction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 