from setuptools import find_packages, setup

# Read requirements from files
with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

with open("requirements-dev.txt") as f:
    # Exclude the '-r requirements.txt' line
    dev_requires = [req for req in f.read().strip().split("\n") if not req.startswith("-r")]

setup(
    name="pipelines",
    packages=find_packages(exclude=["pipelines_tests"]),
    install_requires=install_requires,
    extras_require={"dev": dev_requires},
)
