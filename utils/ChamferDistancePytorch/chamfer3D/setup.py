from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='chamfer_3D',
    version='0.0.0',
    packages=find_packages(),  # 自动找到 chamfer_3D 包
    ext_modules=[
        CUDAExtension('chamfer_3D.chamfer_3D', [  # ← 添加包前缀
            os.path.join('chamfer_3D', 'chamfer_cuda.cpp'),  # ← 修改路径
            os.path.join('chamfer_3D', 'chamfer3D.cu'),
        ],
        extra_link_args=[
            '-Wl,-rpath,$ORIGIN/../torch/lib',  # ← 调整相对路径(多一层目录)
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    package_data={'chamfer_3D': ['*.so']},  # 确保 .so 包含在 wheel 中 (推荐保留)
    zip_safe=False,  # 二进制扩展不能压缩安装
    install_requires=['torch>=2.0.0'],
)