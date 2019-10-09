# Install
python setup.py

# Testing the analysis pipeline
Make sure you have the test data file and run the following commands:
```
python beans rows -m kmeans <test data directory path> <path to an output directory>
```

If the analysis fails and you need to rerun it but want to start over from scratch, include `--reset`

After you get the export, you can add the genotype names to each entry by running:
```
python beans add-genotypes <export file path> <genotypes file path>
```