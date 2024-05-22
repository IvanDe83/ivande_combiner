from setuptools import setup, find_packages

setup(
    name="ivande_combiner",
    version="0.0.10",
    packages=find_packages(),
    description="basic functionality for classic sklearn ml",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="IvanDe",
    author_email="ivande83@gmail.com",
    url="https://github.com/IvanDe83/ivande_combiner",
    license="MIT",
    install_requires=[
        "pytest==8.2.1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
)
