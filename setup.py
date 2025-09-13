from setuptools import setup, find_packages

setup(
    name='re_mcl',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "mif": ["data/*.mtx"],
    },
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'networkx>=2.6',
        "importlib-resources; python_version<'3.9'"
        # These libraries are currently all you need.
    ],
    author='Hiroyuki Akama',
    author_email='akamalab01@gmail.com',
    description='This is a Python program for Markov Clustering (MCL) that supports not only dense matrices (converted to CSR format) but also sparse matrices (can read Matrix Market mtx files).',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/hilolani/re_mcl',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
