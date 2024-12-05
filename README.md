poetry run pytest

poetry run sphinx-apidoc -o docs ./panther

cd docs | .\make.bat clean | .\make.bat html

poetry run pre-commit install
