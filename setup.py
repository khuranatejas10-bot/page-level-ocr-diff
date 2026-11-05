from setuptools import setup, find_packages

setup(
    name="pageocrdiff",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "opencv-python-headless",
        "pillow",
        "scipy",
        "scikit-image",
        "pyyaml",
    ],
)
