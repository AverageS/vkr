import os

from setuptools import setup

reqs_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')

doc = """
Определение авторства публикации в онлайновой социальной сети

Модуль предоставляет интерфейс для определения авторства на основе текстовых публикаций данных из социальных сетей
"""

version = '0.2.6'
url = 'https://test:testtest@dev.mrbot.im/mrbot/mrrest.git'

setup(
    name='sitnikov_vkr',
    version=version,
    maintainer='Alexey Sitnikov',
    maintainer_email='i2rdt@ya.ru',
    packages=['sitnikov_vkr'],
    description=doc.strip().split('\n')[0],
    long_description=doc,
    url=url,
    install_requires=[s.strip() for s in open(reqs_file).readlines()],
    package_data={'sitnikov_vkr': ['data/*', 'classifiers/*']}
)
