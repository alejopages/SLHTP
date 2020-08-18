from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="beanpheno",
    version="0.0.1",
    author="Alejandro D. Pages",
    author_email="apages2@unl.edu",
    description="Bean image analysis tool for phenotyping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alejandropages/SLHTP.git",
    packages=find_packages(),
    install_requires=['click', 'scikit-learn', 'scikit-image', 'pandas', 'numpy', 'matplotlib', 'imagecodecs'],
    entry_points={
        'console_scripts':['beans = beans.beans:start']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
