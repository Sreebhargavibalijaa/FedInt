from setuptools import setup, find_packages

setup(
    name='fedint',
    version='0.1',
    description='Federated Learning Interpretability Tool',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8',
        'scikit-learn',
        'pandas',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
