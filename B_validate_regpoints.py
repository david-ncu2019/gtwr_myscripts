#!/usr/bin/env python
# coding: utf-8

# In[1]:


from appgeopy import *
from my_packages import *


# In[2]:


# geospatial file of MLCW
# showing the location and information
mlcw_gdf = gpd.read_file(
    r"D:\1000_SCRIPTS\003_Project002\20250222_GTWR001\2_KrigingInterpolation\points_fld\mlcw_twd97.shp"
)


# In[3]:


select_regpoints_file = "Layer_4.feather"
regpoints_base = select_regpoints_file.split(".")[0]
layer_from_base = regpoints_base.replace("_", " ")

select_gtwr_csv = r"D:\1000_SCRIPTS\003_Project002\20250222_GTWR001\5_GTWR_Prediction\gtwr_Layer_4_kernel-bisquare_lambda-0d005_bw-17_coefficients.csv"


# In[21]:


# Read the regression point data. This file contains the predicted values at every grid point for every time step.
regpoints_df = pd.read_feather(select_regpoints_file)
regpoints_df = regpoints_df.drop(["id", "prediction_timestamp", "model_name"], axis=1)
# regpoints_df = regpoints_df.set_index("PointKey")


# In[22]:


# To perform a spatial search efficiently, we only need the unique locations of the grid points.
# We can get this by filtering the dataframe for just one time period (e.g., time_period == 1).
print("  - Preparing unique spatial locations for regression points...")
regpoints_df_byPointKey = regpoints_df.query("monthly==1")

# Convert this subset of unique points into a GeoDataFrame.
# This is essential for performing spatial operations like buffering and searching.
regpoints_df_byPointKey = geospatial.convert_to_geodata(
    df=regpoints_df_byPointKey, xcoord_col="X_TWD97", ycoord_col="Y_TWD97", crs_epsg="EPSG:3826"
)


# In[23]:


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Step 2: Load and Prepare the GTWR Model Output (the monitoring stations)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Read the CSV file containing the GTWR model's output.
# This file includes the original ("y") and predicted ("yhat") values at the monitoring station locations.
gtwr_output = pd.read_csv(select_gtwr_csv)

# Create a unique identifier ('PointKey') for each monitoring station based on its coordinates.
# This allows us to easily filter and query data for a specific station.
gtwr_output["PointKey"] = [
    f"X{int(x*1000)}Y{int(y*1000)}" for x, y in zip(gtwr_output["X_TWD97"], gtwr_output["Y_TWD97"])
]

# Get a list of the unique monitoring stations that were processed.
unique_pointkeys = gtwr_output["PointKey"].unique()

# Convert the GTWR output data into a GeoDataFrame to enable spatial analysis.
gtwr_output = geospatial.convert_to_geodata(
    df=gtwr_output, xcoord_col="X_TWD97", ycoord_col="Y_TWD97", crs_epsg="EPSG:3826"
)


# In[31]:


fig_savefld = os.path.join(os.getcwd(), "figure_validate_regpoints")
if not os.path.exists(fig_savefld):
    os.makedirs(fig_savefld)



# select_pointkey = unique_pointkeys[0]
for select_pointkey in tqdm(unique_pointkeys, desc=f"Validating {layer_from_base}", position=1, leave=False):

    # Get the data for the single, current monitoring (MLCW) station.
    mlcw_data_byPointKey = mlcw_gdf.query("PointKey==@select_pointkey")
    mlcw_station_name = mlcw_data_byPointKey.STATION.values[0]
    
    # Get the GTWR model's time-series output for this specific station.
    df_byPointKey = gtwr_output.query("PointKey==@select_pointkey")
    
    # --- Geospatial Search ---
    # Find all the regression grid points that are physically close to the current monitoring station.
    # HINT: The `buffer_radius` is a key parameter. Here it's set to 500 units (likely meters).
    # You can change this value to include more or fewer neighboring points in the analysis.
    search_points_around_mlcw = geospatial.find_point_neighbors(
        central_point=mlcw_data_byPointKey.iloc[0],
        target_points_gdf=regpoints_df_byPointKey,
        central_key_column="STATION",
        buffer_radius=500,
    )
    
    # Get the unique identifiers of these neighboring points.
    points_around_mlcw_byPointKey = search_points_around_mlcw.index
    
    # From the full regression point dataset, extract all time-series measurements for these neighboring points.
    points_around_mlcw_measurements = regpoints_df.loc[points_around_mlcw_byPointKey, :]
    points_around_mlcw_measurements = points_around_mlcw_measurements.sort_values("monthly")
    
    # --- Plotting ---
    # Create a new figure for this station's validation plot.
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    
    # Plot 1: Plot all individual predicted values from the neighboring grid points.
    # These are shown as light grey, semi-transparent circles to visualize the spread of predictions.
    for t in regpoints_df["monthly"].unique():
        points_around_mlcw_measurements_byTime = points_around_mlcw_measurements.query(
            "monthly==@t"
        )
        ax.plot(
            points_around_mlcw_measurements_byTime["monthly"],
            points_around_mlcw_measurements_byTime[f"Predicted_{regpoints_base}"],
            marker="o",
            linestyle=" ",
            markerfacecolor="none",
            markeredgecolor="lightgrey",
            alpha=0.6,
        )
    
    # Plot 2: Calculate and plot the AVERAGE of the predictions from the neighboring points at each time step.
    # This gives a single, smoothed time-series representing the model's general prediction for that local area.
    average_measurements = points_around_mlcw_measurements.groupby("monthly").mean()
    ax.plot(
        average_measurements[f"Predicted_{regpoints_base}"],
        marker="s",
        linestyle="--",
        color="blue",
        alpha=0.5,
        label="Average Predicted",
    )
    
    # Plot 3: Plot the ORIGINAL, measured data from the actual monitoring station.
    # This is the "ground truth" that we are comparing our model's predictions against.
    ax.plot(df_byPointKey["time_stamp"], df_byPointKey["y"], color="magenta", marker="o", label="Original")
    
    # Plot 4 (Optional): Plot the direct GTWR prediction at the station location.
    # You can uncomment this to see how the direct prediction ('yhat') compares.
    # ax.plot(df_byPointKey["time_stamp"], df_byPointKey["yhat"], color="lime", marker="o", label="y_hat", alpha=0.25)
    
    # --- Finalizing and Saving the Figure ---
    
    # Apply custom formatting to the plot axes and title.
    visualize.configure_axis(
        ax=ax, title=f"{mlcw_station_name} - {layer_from_base}", hide_spines=["right", "top"], fontsize_base=12
    )
    
    # Configure the plot legend.
    visualize.configure_legend(ax=ax, fontsize_base=12, labelspacing=0.1, handletextpad=0.2)
    
    fig_outpath = os.path.join(fig_savefld, f"{mlcw_station_name}.png")
    
    # Save the figure with specific dimensions and resolution.
    visualize.save_figure_with_exact_dimensions(fig=fig, savepath=fig_outpath, width_px=2000, height_px=680, dpi=300)
    
    # Close the figure to free up memory before starting the next loop iteration.
    plt.close(fig)
    # plt.show()


# In[ ]:




