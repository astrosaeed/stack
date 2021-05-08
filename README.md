## Download the repo

`$ git clone https://github.com/astrosaeed/stack.git` 
`$ cd stack`

 

## Install libraries and create conda environment
Type `make conda-update` and then activate it by typing `conda activate stack`, finally type in  `make pip-tools`

## Download the data

Either download the data images from [here](https://drive.google.com/drive/folders/11N1g1IlibA0AN-GnYzboqnNM1t4aJjG4?usp=sharing) or you can generate data using the script

### (Optional) generating the data
Download [CoppeliaSimEdu](https://www.coppeliarobotics.com/downloads) and follow the instructions to run it. Assuming you are uisng Ubuntu, run the software using 
`$ cd path_to_CoppeliaSim_folder/`
`$ ./coppeliaSim.sh -h path_to_project_repo/stack/stability_checker/data/coppelia_simulator/stack.ttt` 

and then run 
`$ path_to_project_repo/stack/stability_checker/data/data_collection.py`

## Training
`$ cd stack `
`$ python3 training/run_experiment.py --model_class=MLP --data_class=STACK --max_epochs=5 --gpus=1`

### Experiment management
To do
