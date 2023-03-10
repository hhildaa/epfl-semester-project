# This file includes a function to visualize clusters on a world map.
import plotly.express as px
import pandas as pd
import numpy as np

config = {
  'toImageButtonOptions': {
    'format': 'png', 
    'filename': 'image',
    'height': 500,
    'width': 700,
    'scale':6 
  }
}

def cluster_visualization(df: pd.DataFrame, cluster: np.ndarray, name: str) -> None:
    """
    Plots the given clusters on a choropleth map using the ISO codes from the dataframe.
    
    Args:
        df: The dataframe to use for the map.
        cluster: An array of cluster labels for each sample in the data.
        name: The name of the image file to save the plot to. The file will be saved in the "results/pics" directory.
    
    Returns:
        None
    """
    df_vis = pd.DataFrame()
    df_vis["country_name"] = df["WP5: Country"]
    df_vis["clusters"] = cluster
    df_vis["ISO_code"] = df["COUNTRY_ISO3: Country ISO alpha-3 code"]
    fig = px.choropleth(df_vis, locations=df_vis["ISO_code"],
                        color="clusters",
                        hover_name="country_name",
                        color_continuous_scale=px.colors.sequential.Plasma)
    fig.write_image(f"results/pics/{name}.png")
    fig.show(config = config)