from setuptools import setup, find_packages

print(find_packages(where='deepbc/src'))

setup(
    name='deepbc',
    version='1.0.1',
    url=None,
    author='Klaus-Rudolf Kladny',
    description='Deep Backtracking Counterfactuals',
    packages=find_packages(where='deepbc/src'),
    package_dir={'': 'deepbc/src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)