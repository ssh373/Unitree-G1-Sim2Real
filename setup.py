from setuptools import setup, find_packages

setup(
    name='arc_g1',
    version='25.2.0',
    packages=find_packages(),
    include_package_data=True,
    author="Sol Choi",
    author_email="solchoi@yonsei.ac.kr",
    maintainer="Sol Choi",
    maintainer_email="solchoi@yonsei.ac.kr",
    description="Scripts and utilities for arc_g1",
    url="https://github.com/S-CHOI-S/HumARConoid-Sim2Real.git",
    license="MIT",
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            # 'arc_g1-run=main:main'
        ],
    },
)
