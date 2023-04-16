import pandas as pd
import pydeck as pdk
# import plotly.graph_objs as go 

CHLOROPLETH = 'Chloropleth'
BUBBLE_MAPS = 'Bubble Maps'
DENSITY_HEATMAP = 'Density Heatmap'
SCATTER_MAP = 'Scatter Map'
PYDECK_CHART = 'Pydeck Chart'

def createGeoPlot(type, df, params):
   if type == SCATTER_MAP:
      df_new = pd.DataFrame({'lat': df[params['lat']], 'lon': df[params['lon']]})
      return df_new
   if type == PYDECK_CHART:
      df_new = pd.DataFrame({'lat': df[params['lat']], 'lon': df[params['lon']]})
      return pdk.Deck(
               map_style=None,
               initial_view_state=pdk.ViewState(
               latitude=31.7833,
               longitude=35.2167,
               zoom=11,
               pitch=50),
               layers=[
                  pdk.Layer(
                     'HexagonLayer',
                     data=df_new,
                     get_position='[lon, lat]',
                     radius=200,
                     elevation_scale=4,
                     elevation_range=[0, 1000],
                     pickable=True,
                     extruded=True,
                  ),
                  pdk.Layer(
                        'ScatterplotLayer',
                        data=df_new,
                        get_position='[lon, lat]',
                        get_color='[200, 30, 0, 160]',
                        get_radius=200,
                  ),
               ],
            )
