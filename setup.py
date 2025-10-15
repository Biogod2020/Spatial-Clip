# 文件路径: diffusion_project/git_repo/yuanspace/setup.py

from setuptools import setup, find_packages

setup(
    name="open-clip-torch",
    version="2.20.0-dev",  # 使用一个自定义版本号以示区别
    author="Yuanspace Custom Version",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # 在这里可以添加您项目依赖的包，但对于可编辑安装通常不是必需的
    ],
)
