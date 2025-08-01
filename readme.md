TODO: 
- Change all scripts to output to sim_outputs folder
- Delete all sim outputs and other miscellanea from github
- Make repo public

# Finite Element Methods and Structure Preservation

This repository accompanies my MSc dissertation "Finite Element Methods and Structure Preservation" (link to appear). Most experiments are run using Firedrake, so Firedrake must be installed - details on how to do this are available [here](https://www.firedrakeproject.org/firedrake/install).

## Running Firedrake in Docker container
TODO: TEST THAT THIS IS ACTUALLY TRUE. The way I ran these scripts was using the Firedrake Docker image. To use this, first [install Docker](https://docs.docker.com/desktop/) on your system, and then pull the Firedrake image using
```
> docker pull firedrakeproject/firedrake:latest
```
Then, you can start a container from the image and open the bash terminal inside the container with
```
> docker run -it firedrakeproject/firedrake:latest
```
You must then clone this repository into the docker container
```
# git clone https://github.com/markalvaress/fem_msc_diss.git
```
And from here you can run the experiments.

## Running experiments
All experiments in the paper are contained in the `experiments/` folder. Whether in a Docker container or just on a machine with Firedrake appropriately installed, navigate to the base directory of this project and run an experiment by running the following in the terminal
```
# bash experiments/<experiment_name>.sh
```
If you do not have `bash` installed you can run experiments yourself - inspect the `.sh` file and adapt to your own machine (e.g. on Windows you will likely need `python <script>.py` rather than `python3 <script>.py`).

I SHOULD MAKE SCRIPTS TELL USER WHERE OUTPUT SAVED.

## Other
The `misc/` folder contains various scripts I created when learning to use Firedrake. These are not included in the paper.