from setuptools import setup

setup(name='heatmap',
      version='0.1',
      description='Package to combine image and heat map',
      url='http://github.com/durandtibo/heatmap',
      author='Thibaut Durand',
      author_email='durand.tibo@gmail.com',
      license='MIT',
      packages=['heatmap'],
      install_requires=['numpy', 'scikit-image', 'scipy', 'matplotlib'],
      zip_safe=False)