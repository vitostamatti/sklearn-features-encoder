from setuptools import setup, find_packages


setup(
    name='features_encoder',
    version='0.1.0',
    description='feature encoding transformer',
    # long_description=long_description,
    license="MIT",
    author='Vito Stamatti',
    package_dir={'':'.'},
    packages=find_packages(where='.'),
    install_requires=[
        'scikit-learn', 
    ],
),