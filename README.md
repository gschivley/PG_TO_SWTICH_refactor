# PG_TO_SWTICH_refactor

This code and related settings are designed to create [SWITCH](https://github.com/switch-model/switch) inputs from [Powergenome](https://github.com/PowerGenome/PowerGenome).

Ahead of deeper dive, please make sure you have Powergenome installed in your local machine, follow [Here](https://github.com/PowerGenome/PowerGenome#installation) to get started.

# Running code

Clone this repository to your local machine and copy this folder into your own project folder -- Having your own project folders separate from the cloned folder will make it easier to pull changes as they are released).

## Follow the steps below to get the code run.

1. Change the settings file as your own needs, check the details of settings components in [Wiki](https://github.com/JennyZheng-uh/PG_TO_SWTICH_refactor.wiki.git).
2. Navigate to your project directory and run the command:
```
 python pg_to_switch.py settings_TD_east.yml test_full
```
```settings_TD_east.yml``` and ```test_full``` are the file names for settings file and outputs folder, change the command line if you rename them.
