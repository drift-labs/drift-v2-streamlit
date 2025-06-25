from setuptools import find_packages, setup

setup(
    name="pipelines",
    packages=find_packages(exclude=["pipelines_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud",
        "dagster-aws",
        "boto3",
        "pandas",
        "s3fs",
        "pyarrow"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
