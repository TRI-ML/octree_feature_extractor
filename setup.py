from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

ext_modules = [
    CUDAExtension('ofe.cuda.ofe', [
        'ofe/cuda/ofe_cuda.cpp',
        'ofe/cuda/ofe_cuda_kernel.cu',
    ]),
]

setup(
    description='PyTorch implementation of "Occlusion Feature Extractor (OFE)"',
    author='Shun Iwase',
    author_email='siwase@andrew.cmu.edu',
    license='MIT License',
    version='1.0.0',
    name='ofe_pytorch',
    packages=['ofe', 'ofe.cuda'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
