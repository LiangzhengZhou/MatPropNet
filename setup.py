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
            "matpropnet-embed-vis=matpropnet.cli.embed_vis:main",
            "matpropnet-explain=matpropnet.cli.explain:main",
            "matpropnet-ensemble-explain=matpropnet.cli.ensemble_explain:main",
        ]
    },
    install_requires=[
        "ase>=3.22",
        "lmdb>=1.4",
        "numpy>=1.23",
        "pyyaml>=6.0",
        "scipy>=1.10",
        "sympy>=1.11",
        "torch>=2.0",
        "torch-geometric>=2.3",
        "tqdm>=4.64",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
        "logging": ["tensorboard>=2.12", "wandb>=0.16"],
        "materials": ["pymatgen>=2023.5"],
        "visualization": [
            "matplotlib>=3.7",
            "scikit-learn>=1.3",
            "umap-learn>=0.5",
        ],
    },
)
