
# JSON documents organization

There are a lot of parameters involved in those experiments, so I have to be careful how the data is stored.

## Facets

The data is called in json documents called facets.
A facet has 5 keys:
- *what*. A kind of type of data, that describes what we find in the data array.
- *metadata*. Labels that distinguish this experiment from others.
- *data*. The data. Often it takes the form of an array of dictionaries for each data row, but it could be a matrix or a tensor.
- *statistics*. Information that summarizes the data. It could be a mean, or something like that.
- *facets*. The children of this facet. For instance, *registrations* facets can have *clusterings* facet as children.

## In the code

That means we have to distinguish between 3 big types of data in the python code.
Python functions should describe what they take as input, it would help me a lot.

- *facets*. The full facets described previously.
- *data row*. The contents of the data array of the facets. For instance, in a registration result it contains the transformation matrix plus the time it took to compute it. 
- *pure data*. Often numpy matrices, the data without any other information.

## Important data types

### registrations

A collection of registration results.


#### Metadata

- `algorithm`. The name of the registration algorithm.
- `algorithm_config`. The configuration string of the algorithm.
- `command`. Ran command.
- `dataset`. Name of the dataset used for the experiment.
- `date`. The date the experiment was computed.
- `ground_truth`. 4x4 matrix representing the ground truth transform of both clouds.
- `initial_estimate_mean`. The mean the initial estimates were sampled from.
- `reading`. The index of the reading in the dataset.
- `reference`. The index of the reference in the dataset.
- `var_translation`. The variance applied to the 3 translation components of the initial estimate.
- `var_rotation`. The variance applied to the 3 rotation components of the initial estimate.

#### Statistics

- `n_registrations`. The number of registrations in the data dict.
- `mean`. The average registration result.
- `covariance`. The covariance of registration results.

#### Data row

- `result`. The registration result. A 4x4 matrix.
- `initial_estimate`. The initial estimate the optimization process started from. A 4x4 matrix.
- `time`. the time it took to compute this registration.

#### Facets

### clusterings

#### Metadata

- `algorithm`. Name of the clustering algorithm used.

#### Data row

- `clustering`. The index of the points in the clusters. A list of lists.
- `mean`. The mean of the points in the cluster.
- `covariance`. The covariance of the points in the cluster.
- `dbscan_radius`. The radius used as dbscan parameter.
- `dbscan_dentity`. The radius multiplied by the number of points in the clustered dataset.
- `dbscan_n`. The number of points used in the dbscan algorithm. 

### trails

#### Metadata

#### Data row

- `trail`. The path the optimizer took.
- `time`. The time it took to complete that trail.
