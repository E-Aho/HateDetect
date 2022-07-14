from setuptools import setup

with open("requirements.txt", "r") as file:
    dependencies = file.readlines()

setup(
    name="HateDetect",
    version="0.1",
    author="Erin Aho",
    install_requires=dependencies
)