from setuptools import setup, find_packages

setup(name='webpager',
      version='0.1',
      description="Paginating the web",
      long_description="",
      author="terry",
      author_email="terry@scrapinghub.com",
      install_requires=['scikit-learn', 'lxml'],
      packages=find_packages(),
      package_data = {
          'webpager.models': ['*.pkl'],
      },
)