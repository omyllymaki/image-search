# Image Tools

Misc set of scripts and tools to handle image files, e.g. collected photos.

## Scripts

**create_library**

Create library from figures. Library contains calculated feature vectors and paths to images.

Example usage:

```
python create_library.py -d dataset
```

**cluster_library**

Group library images to clusters where every cluster contains similar images.

Example usage:

```
python cluster_library.py --select_distance --show_examples
```


**image_query**

Given query image, search similar images from library.

Example usage:

```
python image_query.py -q dataset/100000.png -s 0.90
```

