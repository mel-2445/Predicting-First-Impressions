### Predicting First Impressions with Deep Learning
<https://arxiv.org/abs/1610.08119>

## Replicating the Results

### Create a virtual environment

I prefer to have a clean python 3.7 (or whatever version) conda environment to work from.
I activate the conda environment and create a virtual environment for my new project and upgrade pip.

```
$ conda activate py37
(py37) $ python -m venv venv
(py37) $ conda deactivate
$ . venv/bin/activate
(venv) $ pip install --upgrade pip
```

I have no idea why pip always seems to be out of date when I do this, but whatever.

### Extract data

I'm assuming you are using linux (this probably works on mac as well).
If you are on windows, you will have to do this manually.
I think 7zip or WinRar can handle 'tar.gz' archives.

All we are doing here is extracting the data into the folders that work for the provided python files.

```
$ tar -xzf Images.tar.gz
$ tar -xzf Annotations.tar.gz
$ tar -xzf Spaces.tar.gz
```

This will result in 3 new folders with the respective names of the archives.