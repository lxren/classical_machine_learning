from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description="Exploring classical machine learning techniques include Decision Tree, Nearest Neighbour, and Logistic Regression",
    author='Lily Ren',
    license='MIT',
    install_requires=['pandas','numpy','mlcroissant','tensorflow_datasets','itertools']
)