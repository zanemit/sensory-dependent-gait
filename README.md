Code shared in this repository supports the study "Distinct descending and biomechanical influences on interlimb coordination in mice".

Repository structure:
- `processing\` contains Python code used to preprocess the raw data
- `lme\` contains R code for linear and circular-linear regression analyses and circular mixture analyses
- `figures\` contains Python code to recreate figure panels, organised by figure
- `images\` contains png and svg files of the figures

Data to generate the figures will be made available on Figshare at https://doi.org/10.6084/m9.figshare.30735518 after publication.

**To recreate the figures**:
1. Download and unzip all data from the Figshare link above. Due to some long file names, the safest option might be to unzip data programmatically using 7z ([https://www.7-zip.org](https://www.7-zip.org/)).
    - To unzip with 7z, run `7z x "C:\Users\me\Downloads\dataset.zip" -o"D:\sdgait-data"` from the terminal, using the correct paths. The long (~200 characters) file names necessitate that the folder name (and path) is kept relatively short. 
2. Download this repository
3. Navigate to the local repository.
    - Open `processing\data_config.py`. Edit the **root** path (line 5), pointing it to the unzipped data folder.
    - Open `processing\fig_config.py`. Edit the **savefig\_folder** path (line 43), pointing it to the desired folder for generated figures. If the path does not exist, it will be created once any figure is generated.
4. To recreate Figure 1B (for example), run `python -m figures.fig1.B` from the local repository. The generated plot is saved in the defined savefig\_folder.