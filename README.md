1. (Optional) Set up a virtual environment
```
conda create --name myrtle python=3.8
conda activate myrtle
```
2. Clone the linearfits project repo and install the model testbed
```
git clone git@github.com:millerjohnp/harder-or-different.git testbed
pip install -e testbed
```

3. Look at and then run the example script (assumes there is a GPU available)
```
python example.py
```
