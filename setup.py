from setuptools import setup, find_packages

setup(
    name="energy_forecast",  # Επίλεξε ένα όνομα για το package
    version="0.1",
    packages=find_packages(include=['energypackage', 'energypackage.*']),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "pyyaml"
        # ... πρόσθεσε κι άλλες βιβλιοθήκες αν χρειάζεται
    ],
    entry_points={
        "console_scripts": [
            "energy_cli = energypackage.cli:main",
        ],
    },
)
