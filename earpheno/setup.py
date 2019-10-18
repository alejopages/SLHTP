from setuptools import setup, find_packages

setup(
      # mandatory
      name="earpheno",
      # mandatory
      version="0.1",
      # mandatory
      author_email="apages2@unl.edu",
      packages=['earpheno'],
      package_data={},
      install_requires=['pandas', 'click', 'numpy', 'opencv-python'],
      entry_points={
        'console_scripts': ['earpheno = earpheno.earpheno:start']
      }
)