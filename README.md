# AA272-project

Refer to this link for running GNSS Analysis Tool in MATLAB:
https://github.com/google/gps-measurement-tools

## Filter

Main filter is in the gnss-running-filter folder.

The python library is created with poetry. With poetry installed, you can run the following command to install the library while in the folder:

```python
poetry install
```

The main function is in the file `gnss-running-filter/gnss_running_filter/ins_gnss.py`. The variable at the top should be changed to match local setup. To run, use the following command:

```python
poetry run python path/to/ins_gnss.py
```
