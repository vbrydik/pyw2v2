from setuptools import setup, find_packages

VERSION = '0.0.1a1'
DESCRIPTION = 'Simple wav2vec2 wrapper'
LONG_DESCRIPTION = 'A simple wrapper for wav2vec2 for accelerated ASR research.'

# Setting up
setup(
    name="pyw2v2",
    version=VERSION,
    author="vbrydik (Vitalii Brydinskyi)",
    author_email="<vbrydinskyi@gmail.com>",
    url="https://github.com/vbrydik/pyw2v2",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["torch", "torchaudio", "librosa", "transformers", "datasets"],
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