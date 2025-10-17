from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

'''
Before Reinstalling: rm -rf build/ dist/ egu_extension.egg-info/
python setup.py install
'''

ext_modules = [
    CUDAExtension(
        name = 'egu_impl_cuda',
        sources = ['egu_impl_cuda.cu'],
        extra_compile_args = {
            'cxx': ['-O3', '-fPIC', '-std=c++17'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '--fmad=true',
                '-gencode', 'arch=compute_70,code=sm_70',
                '-gencode', 'arch=compute_80,code=sm_80',
                '-gencode', 'arch=compute_86,code=sm_86',
                # 注意：根据GPU型号调整
                '-gencode', 'arch=compute_70,code=compute_70',
                '-gencode', 'arch=compute_80,code=compute_80',
                '-gencode', 'arch=compute_86,code=compute_86',
                '--ptxas-options=-v',
                '-maxrregcount=64',
                '-std=c++17'
            ]
        }
    )
]

setup(
    name = 'egu_extension',
    version = '1.0',
    ext_modules = ext_modules,
    cmdclass = {'build_ext': BuildExtension},
)