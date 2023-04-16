import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

LINE = 'Line'
BAR = 'Bar '
COUNT_PLOT = 'Count Plot'
SCATTER = 'Scatter'
BUBBLE = 'Bubble Plot'
PIE = 'Pie'
BOX = 'Box Plot'
AREA = 'Area'
HISTOGRAM = 'Histogram'
DISTPLOT = 'Dist Plot'

NUMERICAL = 'Numerical'
CATEGORICAL = 'Categorical'

def createFigure(type, df, params):
    fig = None
    values_rejected = ''
    x = params['x']

    if "color" in params:
        color = params['color'] 
    
    if "y" in params:
        y = params['y'] 
    
    if type == LINE:     
        fig = px.line(df, x=x, y=y, color=color, markers=True)

    elif type == BAR:
        color_type = params['color_type']
        if color_type is not None:
            if color_type == NUMERICAL:
                df[color] = pd.to_numeric(df[color], downcast='float')
            else:
                df[color] = pd.Categorical(df[color])

        fig = px.bar(df, x=x, y=y, color=color,
            hover_data=params['hover'][0], 
            barmode=params['barmode'])

    elif type == COUNT_PLOT:
        df[x] = pd.Categorical(df[x])
        if color is None:
            df = df.groupby(by=x).size().reset_index(name="Counts")
        else:
            df[color] = pd.Categorical(df[color])
            df = df.groupby(by=[x, color]).size().reset_index(name="Counts")

        fig = px.bar(df, x=x, y="Counts", color=color, barmode=params['barmode'])
        fig.update_xaxes(type='category')
        fig.update_xaxes(categoryorder='category ascending')

    elif type == HISTOGRAM:
        fig = px.histogram(df, x=x, color=color, marginal="box")

    elif type == DISTPLOT:
        fig = px.histogram(df[x])

    elif type == PIE:
        fig = px.pie(df, values=x, names=color)
   
    elif type == SCATTER:
        size = None if params['size'][0] == None else df[params['size'][0]]
        fig = px.scatter(df, x=x, y=y, color=color,
                size=size, 
                hover_data=params['hover'][0])
        
    elif type == BUBBLE:
        size_max = None if params['size_max'] == 0 else params['size_max']
       
        fig = px.scatter(df, x=x, y=y, color=color,
                size=df[params['size'][0]], 
                hover_name=params['hover'][0],
                log_x=params['log_x'],
                size_max=size_max)

    elif type == BOX:
        fig = px.box(df, x=x, y=y, color=color, 
                    notched=params['notched'],
                    hover_data=params['hover'],
                    points=params['points'].lower())

    elif type == AREA:
        fig = ''
        # fig = px.area(df, x=x, y=y, color=color, line_group=params['line_group'])
    return fig


    
       

