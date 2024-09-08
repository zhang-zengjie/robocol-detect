from setuptools import setup, find_packages

setup(
    name='robocol-detect',
    version='0.1.0',
    description='Collision Detection Using Supervised Learning',
    url='https://github.com/zhang-zengjie/robocol-detect.git',
    author='Zengjie Zhang',
    author_email='z.zhang3@tue.nl',
    license='BSD3',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scikit-learn',
        'matplotlib',
        'numpy'
    ],
    extras_require={
    },
    classifiers=[
    ],
    python_requires='<=3.11',
    entry_points={
    },
    include_package_data=True
)