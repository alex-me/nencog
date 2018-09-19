# nencog project

## sources maintained by alex


## utilities for managing the LAVA dataset

### Lava Corpus
Language and Vision Ambiguities (LAVA) is a multimodal corpus that supports the
study of ambiguous language grounded in vision, see Yevgeni Berzak, Andrei
Barbu, Daniel Harari, Boris Katz and Shimon Ullman (2015). *Do You See What I
Mean? Visual Resolution of Linguistic Ambiguities* for reference

*	[labeled_images.py](./labeled_images.py) is used to list the images with
 	labels from the original json file of the dataset

### Sample extraction
All useful images hase been manually annotated for the presence of significant
objects, together with their embedding box, data er in
[coords.txt](./coords/coords.txt)

*	[put_boxes.py](./put_boxes.py) scan all images annotated in
[coords.txt](./coords/coords.txt) and generated the corresponding images with
boxes in overlay

*	[coord_stat.py](./coord_stat.py) produces a statistics of the boxes
embedding objects in the vairous categories

*	[extract_samples.py](./extract_samples.py) extract all samples

*	[extract_no_samples.py](./extract_no_samples.py) extract samples that do
not contain objects of hte relevant categories, to be used by the neural
segmentation
