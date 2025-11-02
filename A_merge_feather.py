from appgeopy import *
from my_packages import *

prediction_output_folder = r"D:\1000_SCRIPTS\003_Project002\20250222_GTWR001\8_GNNWTR\predictions_20250709_100750_Layer_4"
layer = "Layer_4"
output_files = glob(os.path.join(prediction_output_folder, "*.feather"))

combined_output_df = pd.concat([pd.read_feather(f) for f in tqdm(output_files)])
# save combined dataframe to feather for later use
combined_output_df.to_feather(os.path.join(os.path.dirname(prediction_output_folder), f"{layer}.feather"))