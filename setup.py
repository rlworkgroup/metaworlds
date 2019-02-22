from setuptools import find_packages
from setuptools import setup

# Required dependencies
required = [
    # Please keep alphabetized
    'cached_property',
    'gym',
    'numpy',
    'joblib',
    'mako',
    'matplotlib',
    'pygame',
    'python-dateutil',
    'scipy',
]

# Framework-specific dependencies
extras = {
    'mujoco': [
        'mujoco-py<1.50.2,>=1.50.1',
        'lxml',
        'PyOpenGL',
    ],
    'box2d': ['box2d-py>=2.3.4'],
}
extras['all'] = list(set(sum(extras.values(), [])))

# Development dependencies (*not* included in "all")
extras['dev'] = [
    # Please keep alphabetized
    'coverage',
    'flake8',
    'flake8-docstrings',
    'flake8-import-order',
    'pep8-naming',
    'pre-commit',
    'pylint',
    'pytest',
    'pytest-cov',
    'sphinx',
    'yapf',
]

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='metaworlds',
    version='0.0.1dev',
    author='Reinforcement Learning Working Group',
    description='Environments for benchmarking meta-learning and multi-task learning',
    url='https://github.com/rlworkgroup/metaworlds',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,
    extras_require=extras,
    license=license,
    long_description=readme,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
    ],
)
