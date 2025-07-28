from setuptools import setup

setup(
    name='museeegprep',
    version='0.1.0',
    py_modules=[
        'prep_config',
        'prep_io',
        'prep_preprocessing',
        'prep_features',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'mne',
        'matplotlib',
    ],
    author='Python course',
    description='Modular Muse-S EEG Preprocessor',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
) 