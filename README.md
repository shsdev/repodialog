# repodialog

## Overview

`repodialog` allows users to ask questions in relation to PDF documents. 

It is available in form of a web application (streamlit) and a Jupyter notebook. 

## Requirements

Configuration and setup tested on Ubuntu 22.04.3 LTS with the following components:

- Python 3.10.12

- Streamlit, version 1.18.1

See the `requirements.txt` file for other dependencies.

You need a Huggingface account to get a Huggingface API key (configure in .env).

## Project setup

Clone the project:

git clone https://git-service.ait.ac.at/dil-demos/repodialog.git

Create a .env configuration file from the example:

```
cp .env.example .env
```

Adapt variables in the .env file.

Create virtual environment:

```bash
virtualenv -p python3 venv
```

and activate it:

```bash
source venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

### Run

#### Run streamlit app

You can run the application using `streamlit`:

```
streamlit run app.py
```

#### Run jupyter notebook

To run the jupyter notebook, install it first if it is not available:

```
pip install notebook
```

Run the jupyter notebook:

```
jupyter notebook repodialog.ipynb
```

