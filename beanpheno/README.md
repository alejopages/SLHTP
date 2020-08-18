# Dependencies
> Python 3.7
Windows, Mac or Linux
Developed on Windows 10

# Installation Instructions
Clone the repository, navigate to the repository and install.

```
git clone https://github.com/alejandropages/SLHTP.git
cd SLHTP/beanpheno
python setup.py develop
```

# Usage Instructions

### 1. Running the Pipeline

Running this pipeline is a hands on process allowing you to test it out on images wwith different background colors, object colors and lighting.  The original intention of this project was to make a fully automated pipeline but the methods implemented required more fine tuning and preprocessing than time allowed. Therefore, I am including this usage instructions section to help you understand how to tweak paramaters during the analysis.

After installation, you will have a command "beans" that can be run from anywhere in your system (there might be issues if you try to run it from within the source directory itself). For help run `beans --help`. There are only two command, add-genotypes and rows. The command that runs the analysis pipeline is `rows`. Run `beans rows --help` for a list of arguments and options.

One of the options that is required is the `--method` option. I had most success with the kmeans method, its the slowest, but by far more effective than the other two. Bare in mind that this was a prototype package, meaning that some of the features were implemented to test an idea, and remained only as an artifact.

Example command:
```
beans rows -m kmeans <path_to_images> <path_to_output>
```

The other option that you may need to use is the `--reset` option. This option deletes the stored parameters and figures for a previous run which results in you being able to start the analysis over from scratch. Its important if you look back and realize that there were errors in your initial pipeline and need to restart. If you only want to redo the analysis on a specific image, you can go through the cache file created and delete the cached parameter and firgure files. There are two cache file directories that will be created in your output dir, "temp" and "pickles". Cached files are labelled with the name of the image for the analysis they are caching. You simply need to delete the files corresponding to the image you want to re-analyze.

I wouldd't mess with the --n_clusters unless you know for sure based on past runs what has worked the best. It is already set to a senseible default of 4 that has worked even for multicolored samples. The fewer the number of clusters, the faster the runtime, howwever the less precise the segmentation of your objects will be.

### 2. Tuning the Parameters

This section applies to the KMeans segmentation method. 

<TODO: Fill in after obtaining images and running tests>
 


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
