# Dependencies
- Python 3.7
- git
- OS: Windows, Mac or Linux

# Installation Instructions

### Windows, Max, Linux

Clone the repository, navigate to the repository and install.

```
git clone https://github.com/alejandropages/SLHTP.git
cd SLHTP/beanpheno
python setup.py develop
```

### Notes on Windows Installation
You may need to set the installation directory and add it to PATH. The install directory should be something out of the way as you won't need to access the files in the directory. 

```
python setup.py develop --install-dir=<path-to-installation-dir>
```
Then add the install directory path to the User PATH environment variable. 

# Usage Instructions

### 1. Running the Pipeline

The original intention of this project was to make a fully automated pipeline but the methods implemented required more development time for fine tuning and preprocessing method implementation than time allowed. Therefore, I am including this usage instructions section to help you understand how to tweak paramaters during the analysis.

After installation, you will have a command "beans" that can be run from anywhere in your system (there might be issues if you try to run it from within the source directory itself). For a list of commands run `beans --help`. There are only two command, add-genotypes and rows. The command that runs the analysis pipeline is `rows`. Run `beans rows --help` for a list of arguments and options.



One of the options is the `--method` option. It is set to kmeans by default so if you would like to use the kmeans method, simply don't include this option. Please note that the kmeans method was by far the most effective method and I wouldn't suggest using the other methods. Bare in mind that this was a prototype package, meaning that some of the features were implemented to test an idea, and remain only as artifacts.

E.g.
```
beans rows <path_to_images> <path_to_output> # will run a kmeans analysis
```

The other option that you may need to use is the `--reset` option. This option deletes the stored parameters and figures for a previous run so you can start the analysis over from scratch. Its important if you look back and realize that there were errors in your initial pipeline and need to restart. If you only want to redo the analysis on a specific image, you can go through the cache file created and delete the cached parameter and figure files. There are two cache file directories that will be created in your output dir, "temp" and "pickles". Cached files are labelled with the name of the image they are associated with. You simply need to delete the files corresponding to the image you want to re-analyze.

If you find a set of parameters that work for all the images in your dataset, and you would like to run the pipeline automatically and simply verify the results in the ouput folder, than you can use the `--auto` option. This option switches off user prompts and does not display images. You can manually set the default selem and n_clusters parameters using the --selem and --n_clusters options.

E.g.
```
beans rows --auto --selem=14 --n_clusters=3 <input-dir> <output-dir> # will run a kmeans analysis without user prompts using the provided params.
```

### 2. Tuning the Parameters

This section applies to the KMeans segmentation method. 

- n_clusters
The number of clusters, or K in the K Means method, effects the precision of the segmentation. Higher K means more precision, lower K means lower precision.

< show example image of low K, then high K >

< show example image of when background color group includes foreground regions >

- selem
The selem is the size of the matrix used to remove speckles from the image. A larger selem will remove larger speckles but may end up eating away at the target objects in your image. 

< show example image of low selem, then high selem. >

# Testing the analysis pipeline
Make sure you have the test data file and run the following commands:
```
python <PATH TO beans.py> rows -m kmeans <test data directory path> <path to an output directory>
```

If the analysis fails and you need to rerun it but want to start over from scratch, include `--reset`

After you get the export, you can add the genotype names to each entry by running:
```
python beans add-genotypes <export file path> <genotypes file path>
```
