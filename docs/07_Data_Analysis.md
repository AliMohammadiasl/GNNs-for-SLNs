# 07 Data Analysis

## Purpose
This notebook performs topological analysis on the SLNs to evaluate the intrinsic structure of the classrooms without passing them through deep learning models.

## How it Works
The script utilizes NetworkX to analyze metrics like the largest connected component size, average graph density, and algebraic connectivity (the Fiedler value). These metrics correlate to how dense, communicative, and robust the student interactions are within a given academic timeline.

## Inputs
- `.bin` graph files or raw dataset files representing the SLN topologies at different subsets of the course timeline.

## Outputs
- `average_degrees.csv` and standard output detailing the sizes of connected components and network connectivity at differing temporal slices.

## Execution
Execute the cells to view the algebraic characteristics. Ensure NetworkX and Pandas are installed in the environment.