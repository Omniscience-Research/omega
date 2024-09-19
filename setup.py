from setuptools import setup, find_packages

setup(
    name="omega-ml",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scikit-learn",
        "scipy",
    ],
    author="Jeremy Nixon",
    author_email="jeremy@omniscience.tech",
    description="A collection of generated machine learning algorithms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/omniscience-research/omega",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)