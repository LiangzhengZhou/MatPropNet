from setuptools import find_packages, setup


setup(
    name="matpropnet",
    version="0.2.0",
    description="General materials property prediction framework with reusable CLI entrypoints",
    url="https://github.com/bt403/ocp-gemnet-gnn",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "matpropnet-preprocess=matpropnet.cli.preprocess:main",
            "matpropnet-train=matpropnet.cli.train:main",
            "matpropnet-eval=matpropnet.cli.eval:main",
            "matpropnet-predict=matpropnet.cli.predict:main",
        ]
    },
)
