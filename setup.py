import setuptools

with open("tella/__version__.py") as fp:
    ns = {}
    exec(fp.read(), ns)

with open("README.md") as fp:
    long_description = fp.read()

requirements = []
with open("requirements.txt", "rt") as f:
    for req in f.read().splitlines():
        if req.startswith("git+"):
            pkg_name = req.split("/")[-1].replace(".git", "")
            if "#egg=" in pkg_name:
                pkg_name = pkg_name.split("#egg=")[1]
            requirements.append("%s @ %s" % (pkg_name, req))
        else:
            requirements.append(req)

setuptools.setup(
    name="tella",
    version=ns["__version__"],
    description="library for continual reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.7",
    packages=setuptools.find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black==22.1.0"],
        "atari": ["gym[atari,accept-rom-license]"],
    },
)
