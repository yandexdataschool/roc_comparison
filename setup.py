import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='roc_comparison',
    version='0.0.1',
    author='kazeevn',
    author_email='unknown',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/yandexdataschool/roc_comparison',
    project_urls = {
    },
    license='MIT',
    packages=['roc_comparison'],
    install_requires=['pandas','numpy','scipy.stats'],
)
