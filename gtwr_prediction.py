#!/usr/bin/env python
# coding: utf-8

# In[1]:


from appgeopy import *
from my_packages import *


# ==============================================================================
# USER CONFIGURATION - Modify these values
# ==============================================================================

FILES = glob(os.path.join(r"D:\1000_SCRIPTS\003_Project002\20250222_GTWR001\8_GNNWTR", "L*.feather"))

for GTWR_FILE in FILES[:1]:
    GTWR_FILE = r"D:\1000_SCRIPTS\003_Project002\20250222_GTWR001\5_GTWR_Prediction\3__PredictionOutput\Layer_1.feather"
    # File paths
    # GTWR_FILE = "3__PredictionOutput/Layer_1.feather"  # Path to your data file
    GTWR_LAYER = os.path.basename(GTWR_FILE).split(".")[0]
    
    STATIONS_SHP = r"D:\1000_SCRIPTS\003_Project002\20250222_GTWR001\2_KrigingInterpolation\points_fld\mlcw_twd97.shp"
    
    # Output settings
    WIDTH_PX = 1709  # 2091
    HEIGHT_PX = 1709  # 2005
    DPI = 300
    
    # Analysis settings
    START_DATE = pd.Timestamp(year=2016, month=5, day=1)
    
    # ==============================================================================
    # PLOT CONFIGURATIONS
    # ==============================================================================
    
    CONFIGS = {
        "pred_CUMDISP": {
            "cmap": "turbo",
            "title": "InSAR",
            "label": "Cumulative Displacement (mm)",
        },
        "prediction": {
            "cmap": "turbo",
            "title": "Predicted MLCW",
            "label": "Cumulative Displacement (mm)",
        },
        "prediction_std": {
            "cmap": "jet",
            "title": "Std. Dev.",
            "label": "Standard Deviation (mm)",
        },
        "coef_CUMDISP": {"cmap": "coolwarm", "title": "Coefficient", "label": None},
        "bias": {
            "cmap": "PiYG_r",
            "title": "Intercept",
            "label": None,
        },
    }
    
    # What to plot
    # QUANTITY = "CUMDISP_coef"  # Options: "CUMDISP_measure", "MLCW_pred", "CUMDISP_coeff", "Intercept"
    
    
    # ==============================================================================
    # MAIN SCRIPT
    # ==============================================================================
    
    # Load data
    print(f"Loading data from {GTWR_FILE}...")
    data = pd.read_feather(GTWR_FILE)
    stations = gpd.read_file(STATIONS_SHP)
    
    # for QUANTITY in ["coef_CUMDISP", "bias", "prediction"]:
    for QUANTITY in ["prediction"]:        
    
        # FIG_SAVEFOLDER = os.path.join(
        #     os.getcwd(), "figure_map_quantity", f"{QUANTITY}", f"{GTWR_LAYER}"
        # )

        FIG_SAVEFOLDER = os.path.join(
            os.path.dirname(GTWR_FILE), "figure_map_quantity", f"{QUANTITY}", f"{GTWR_LAYER}"
        )

        if not os.path.exists(FIG_SAVEFOLDER):
            os.makedirs(FIG_SAVEFOLDER)
            
        # for TIME_PERIOD in trange(1, 67):
        for TIME_PERIOD in [66]:            
            # for TIME_PERIOD in [66]:
            # Filter for selected time period
            plot_data = data.query("monthly == @TIME_PERIOD")
            if plot_data.empty:
                available_periods = sorted(data["monthly"].unique())
                raise ValueError(
                    f"Time period {TIME_PERIOD} not found. Available: {available_periods}"
                )
    
            # Convert to GeoDataFrame
            gdf = convert_to_geodata(plot_data, "X_TWD97", "Y_TWD97", "EPSG:3826")
    
            # Get plot configuration
            if QUANTITY not in CONFIGS:
                raise ValueError(
                    f"Invalid quantity '{QUANTITY}'. Available: {list(CONFIGS.keys())}"
                )
            config = CONFIGS[QUANTITY]
    
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 10))
    
            if QUANTITY == "predict_se":
                cbar_vmin = 3
                cbar_vmax = 8
            else:
                cbar_vmin = -30
                cbar_vmax = 0
                # cbar_vmin = plot_data[QUANTITY].quantile(0.01)
                # cbar_vmax = plot_data[QUANTITY].quantile(0.99)
            # Plot data points
            mappable = spatial_plot.point_values(
                gdf=gdf,
                value_column=QUANTITY,
                ax=ax,
                cmap=config["cmap"],
                edgecolors="none",
                s=15,
                show_colorbar=False,
                vmin=cbar_vmin,
                vmax=cbar_vmax,
            )
    
            # Add stations and basemap
            spatial_plot.show_points(
                stations,
                ax=ax,
                marker="^",
                markersize=40,
                color="black",
                edgecolors="white",
                linewidths=1,
                alpha=1,
            )
            spatial_plot.add_basemap(ax, crs="EPSG:3826")
    
            # Configure appearance
            visualize.configure_axis(ax, tick_direction="out", fontsize_base=14)
            visualize.configure_ticks(
                ax, x_major_interval=20e3, y_major_interval=20e3
            )
    
            # Add colorbar
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, aspect=25, pad=0.025)
            if config["label"]:
                cbar.set_label(config["label"], fontsize=16, fontweight="bold")
            cbar.ax.tick_params(labelsize=14)
    
            # Add annotations
            current_time = START_DATE + relativedelta(months=int(TIME_PERIOD))
            time_str = current_time.strftime("%Y-%m-%d")
    
            ax.text(
                0.05,
                0.95,
                time_str,
                fontsize=20,
                fontweight="bold",
                transform=ax.transAxes,
            )
    
            ax.text(
                0.70,
                0.95,
                GTWR_LAYER.replace("_", " "),
                fontsize=20,
                # ha="right",
                fontweight="bold",
                transform=ax.transAxes,
            )
    
            ax.text(
                0.95,
                0.05,
                config["title"],
                fontsize=20,
                fontweight="bold",
                ha="right",
                transform=ax.transAxes,
            )
    
            # Save figure
            # fig.set_size_inches(WIDTH_PX / DPI, HEIGHT_PX / DPI)
            OUTPUT_FILEPATH = os.path.join(
                FIG_SAVEFOLDER, f"{str(TIME_PERIOD).zfill(3)}.png"
            )
            visualize.save_figure_with_exact_dimensions(
                fig,
                savepath=OUTPUT_FILEPATH,
                width_px=WIDTH_PX,
                height_px=HEIGHT_PX,
                dpi=DPI,
            )
            # fig.savefig(OUTPUT_FILEPATH, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            # plt.show()
    
            # print(f"Plot saved as {OUTPUT_FILEPATH}")
            # print(f"Plotted {QUANTITY} for time period {TIME_PERIOD} ({time_str})")


# In[ ]:





# In[ ]:




