import setuptools      # 导入setuptools, 基于setuptools模块进行打包分发

# 将readme文件中内容加载进来，作为对包的详细说明（可以不需要）

# 调用setuptools的setup进行打包，通过参数配置指定包的信息，这是打包的关键设置
setuptools.setup(
    name="DepthEst", # 这是该包的名字，将来可能使用pip install 该包名直接下载
    version="0.0.1",   #版本号，
    author="Eosin",  #作者
    author_email="ysaipro6@gmail.com", # 作者邮箱
    description="", # 包简短的描述
    # url="https://github.com/pypa/my_pkg",  # 可以将项目上传到github,gitlab等，在此指定链接地址以供下载。

    # 指定需要打包的内容，输入需要打包包名字符串列表，打包时不会自动获取子包，需要手动指定，例如：["my_pkg", "mypkg.utils"]
    packages=['DepthEst','DepthEst.lib'], # 使用该函数可以自动打包该同级目录下所有包
    
    classifiers=[    # 指定一些包的元数据信息，例如使用的协议，操作系统要求
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',   # 该包的Python版本要求
)

