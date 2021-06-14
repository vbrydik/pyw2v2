from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'Simple wav2vec2 wrapper'

with open("./README.md", 'r') as f:
    long_description = "\n" + "".join(f.readlines())

# Setting up
setup(
    name="pyw2v2",
    version=VERSION,
    author="vbrydik (Vitalii Brydinskyi)",
    author_email="<vbrydinskyi@gmail.com>",
    url="https://github.com/vbrydik/pyw2v2",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "torch", 
        "torchaudio", 
        "librosa", 
        "transformers", 
        "datasets", 
        "easydict", 
        "PyYAML",
        "jiwer"
    ],
    keywords=['python'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)