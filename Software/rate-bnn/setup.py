import setuptools

with open("README.md", "r") as f:
	long_description = f.read()

setuptools.setup(
	name="rate",
	version="0.0.1",
	author="Jonathan Ish-Horowicz",
	author_email="jonathan.ish-horowicz17@imperial.ac.uk",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/lorinanthony/RATE",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)