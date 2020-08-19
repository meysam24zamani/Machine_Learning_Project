Course project - ML (MIRI)
Authors:
 - Marti Cardoso 
 - Meysam Zamani

# Documentation files #

- Report.pdf: Report of our ML project, main document.
- Appendix.pdf: Complementary file to the report, it contains figures related to the report.

# Source files information #

All the source R scripts are inside the /source folder. In it we can find the following scripts: 
- 1.PreProcessing.R: Script that runs all preprocessing of our project and saves the resulting datasets in the dataset folder
- 2.SplitTrainTest.R: Script that splits our datasets into train and test and standardizes the continuous variables.
- 3.VisualizationAndFeatureExtraction.R: Script used for the visualization of our dataset (and the creation of D3 and D4).
- 4.ModelSelection.R: Script that runs our model selection task (tries several models). ## WARNING: This script could take more than 2 days to be executed
- modelSelectionUtils.R: Additional file used in 4.ModelSelection.R.
- 5.FinalModel.R: Script that creates our final model and predicts the test set. 

Before running the scripts, you should set the working directory to source/

The source files should be run in the following order: 
 1.PreProcessing.R, 2.SplitTrainTest.R, 3.VisualizationAndFeatureExtraction.R, 4.ModelSelection.R and 5.FinalModel.R
