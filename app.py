import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import functions.charts as ch
import functions.geo_plot as geo
import functions.ml as ml
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

st.write("""
<style>
 :root {
      --main-color: #277BC0;
   }

   h2 {
      color: rgb(13,19,76)
   }

   .title_section {
      color: var(--main-color);
   }

   .filter_num {
      display: inline-block;
      background-color: var(--main-color);
      padding: 4px 8px;
      border-radius: 3px;
      color: white;
      font-weight: 500;
      letter-spacing: 1px;
   }
 
   div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
      box-shadow: 0 0 10px #E5F4F9;
      border: 1px solid #E7F6F2;
      border-radius: 22px;
      padding: 20px 18px;
      background: #FFF
   }

   div[data-testid="stExpander"] {
      background: #FFF
   }

   data-testid="stDataFrameResizable" {
      background: #FFF

   }
 
</style>
""", unsafe_allow_html=True)

LINE = 'Line'
BAR = 'Bar '
COUNT_PLOT = 'Count Plot'
SCATTER = 'Scatter'
BUBBLE = 'Bubble Plot'
PIE = 'Pie Chart'
BOX = 'Box Plot'
HISTOGRAM = 'Histogram'
DISTPLOT = 'Dist Plot'
GEO = 'Geographical Plot'
AREA = 'Area'
chart_types = [BAR, BOX, BUBBLE, COUNT_PLOT, DISTPLOT, GEO, HISTOGRAM, LINE, PIE, SCATTER]
CHLOROPLETH = 'Chloropleth'
BUBBLE_MAPS = 'Bubble Maps'
DENSITY_HEATMAP = 'Density Heatmap'
SCATTER_MAP = 'Scatter Map'
PYDECK_CHART = 'Pydeck Chart'
geo_types = [BUBBLE_MAPS, CHLOROPLETH, DENSITY_HEATMAP, PYDECK_CHART, SCATTER_MAP]

CONTAINS = 'contains'
STARTS_WITH = 'startswith'
ENDS_WITH = 'endswith'
operators = {
   "numerical":['==', '>', '<', '>=', '<='],
   "categorical":['==', CONTAINS, STARTS_WITH, ENDS_WITH]
   }
AND = 'And'
OR = 'Or'
KEEP = 'Keep'
DROP = 'Drop'
REPLACE = 'Replace'
MEAN = 'Mean'
DATA = 'data'
OPERATOR = 'operator'
VALUE = 'value'
NA = 'na'
LOGIC = 'logic'
FILL = 'Fill'
logics = [AND, OR]
na_state = [DROP, REPLACE]
na_replacement = [FILL, MEAN]
COLUMNS = "Columns"
RENAME_COL = 'Rename Column'
DROP_COL = 'Drop Column'
SUBSET = 'Subset Data'
OPERATION = 'Operation'
PERCENTAGE = 'Percentage (%)'
ADDITION = '+'
SOUSTRACTION = '-'
MULTIPLICATION = '*'
DIVISION = '÷'
GAPMINDER = 'Gapminder'
CARSHARE = 'Careshare'
IRIS = 'Iris'
MEDALS_LONG = 'Medals long'
STOCKS = 'Stocks'
TITANIC = 'Titanic'
MEDALS_WIDE = 'Medals wide'
WIND = 'Wind'
TIPS = 'Tips'
ELECTION = 'Election'
EXPERIMENT = 'Experiment'
LINEAR = 'Linear Regression'
LOGISTIC = 'Logistic Regression'
DECISION_TREE = 'Decision Trees'
RANDOM_FOREST = 'Random Forest'
KNN = 'K Nearest Neighbor (KNN)'
SVM = 'Support Vector Machine (SVM)'
PROPHET = 'Prophet - Automatic Forecasting Procedure'
actions = [None, COLUMNS, OPERATION, MEAN, PERCENTAGE]
action_columns = [RENAME_COL, DROP_COL, SUBSET]
action_operations = [ADDITION, SOUSTRACTION, MULTIPLICATION, DIVISION]
datasets = [GAPMINDER, CARSHARE, ELECTION, EXPERIMENT, IRIS,
             MEDALS_LONG, STOCKS, TIPS, TITANIC]
# ml_models = [DECISION_TREE, KNN, LINEAR, LOGISTIC, PROPHET, RANDOM_FOREST, SVM]
ml_models = [DECISION_TREE, KNN, LINEAR, LOGISTIC, RANDOM_FOREST, SVM]
filename = 'dataset'

# @st.cache
def load_csv(file):
   return pd.read_csv(file)

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

def load_dataset(dataset):
   if dataset == GAPMINDER:
      return px.data.gapminder()
   elif dataset == IRIS:
      return px.data.iris()
   elif dataset == MEDALS_LONG:
      return px.data.medals_long()
   elif dataset == MEDALS_WIDE:
      return px.data.medals_wide()
   elif dataset == STOCKS:
      return px.data.stocks()
   elif dataset == TIPS:
      return px.data.tips()
   elif dataset == WIND:
      return px.data.wind()
   elif dataset == CARSHARE:
      return px.data.careshare()
   elif dataset == ELECTION:
      return px.data.election()
   elif dataset == EXPERIMENT:
      return px.data.experiment()
   else:
      return pd.read_csv(f'{dataset}.csv', encoding= 'unicode_escape')

def visualizeNaValues(df):
   fig= px.imshow(df.isna(), aspect='auto')
   return fig

def filterNaValues(df, col, na):
   df_to_filter = df.copy()

   if na != KEEP:
      if na == DROP:
         if col is not None:
            df_to_filter.dropna(subset=[col], inplace = True)
         else:
            df_to_filter.dropna(inplace = True)
      if na == MEAN:
         df_to_filter[col].fillna(df_to_filter[col].mean(), inplace = True)
      # If NA == Replace
      else:
         if col is not None:
            df_to_filter[col].fillna(na, inplace = True)
         else:
            df_to_filter.fillna(na, inplace = True)

   return df_to_filter

def populateParams(type, df):
   col1, col2, col3 = st.columns([1,1,1])
   cols = [None] + df.columns.to_list()
   params = {}

   if (type == BAR) | (type == LINE) | (type == SCATTER) | (type == BUBBLE) \
      | (type == BOX) | (type == AREA):
      params = {
         "x": col1.selectbox('X axis', cols),
         "y": col2.selectbox('Y axis', cols),
         "color": col3.selectbox('Group', cols)
      }

   if (type == SCATTER) | (type == BUBBLE):
      col1, col2, col3 = st.columns([1,1,1])
      params['size'] = col1.selectbox('Size', cols),
      params['hover'] = col2.multiselect('Hover', cols)
      
   if type == BAR:
      col1, col2, col3 = st.columns([1,1,1])
      params['hover'] = col2.multiselect('Hover', df.columns),
      params['barmode'] = col1.radio('Barmode', ['stack', 'group'])
      params['color_type'] = col3.selectbox('Parse Group Type',
                  [None, 'Numerical', 'Categorical'])

   elif type == COUNT_PLOT:
      col1, col2, col3 = st.columns([1,1,1])
      params = {
         "x": col1.selectbox('X axis', cols),
         "color": col2.selectbox('Group', cols),
         "barmode": col3.radio('Barmode', ['stack', 'group'])
      }
            
   elif type == BUBBLE:
      params['size_max'] = col1.number_input('Size Max')
      params['log_x'] = col2.radio('Log x', [True, False])
   
   elif type == BOX:
      params['hover'] = col1.multiselect('Hover', cols)
      params["notched"] = col2.radio('Notched', [False, True])
      params["points"] = col3.radio('Points', ['Outliers', 'All'])

   elif type == PIE:
      col1, col2, = st.columns([1,1])
      params = {
         "x": col1.selectbox('Values', cols),
         "color": col2.selectbox('Names', cols)
      }
   elif type == HISTOGRAM:
      col1, col2 = st.columns([1,1])
      params = {
         "x": col1.selectbox('X axis', cols),
         "color": col2.selectbox('Group', cols),
      }

   elif (type == SCATTER_MAP) | (type == PYDECK_CHART):
      col1, col2 = st.columns([1,1])
      params = {
         "lon": col1.selectbox('Lon', cols),
         "lat": col2.selectbox('Lat', cols),
      }
   # elif type == PYDECK_CHART:
   #    col1, col2 = st.columns([1,1])
   #    params = {
   #       "from": col1.selectbox('From', cols),
   #       "to": col2.selectbox('Lat', cols),
   #    }


   # elif type == AREA:
   #    params['line_group'] = col1.selectbox('Line Group', cols)
    
   return params

def populateFiltersView( df, operators, logics, createFilters, filters):
   for num in range(1, st.session_state["num_filters"] + 1):
      st.markdown(f"<p class='filter_num'>Filter #{num}</p>", unsafe_allow_html=True)

      cols = [None] + df.columns.to_list()
      if num == 1:
         filters.append(createFilters(df, num, cols, operators))
      else:
         filters.append(createFilters(df, num, cols, operators, logics))

def createFilters(df, key, cols, ops, log=None):
   logic = None
   value = None

   if key > 1:
      col1, col2 = st.columns([.2, 1])
      logic = col1.selectbox("And/Or", log, key=str(key)+'_andor')

   col3, col4, col5 = st.columns([1,.5, 1])
   data = col3.selectbox("Data", cols, key=str(key)+'_data')
   operator = None

   if data:
      if df[data].dtype.kind in 'iuf':
         operator = col4.selectbox("Operation", ops['numerical'], key=str(key)+'_op')
         value = col5.number_input('Value', key=str(key)+'_value')
      else:   
         operator = col4.selectbox("Operation", ops['categorical'],
                           key=str(key)+'_op')
         if (operator == CONTAINS) | (operator == STARTS_WITH) | (operator == ENDS_WITH) :
            value = col5.text_input("Text", key=str(key)+'_text')
         else:
            value = col5.selectbox("Value", pd.unique(df[data]), key=str(key)+'_value') 

   return {DATA:data, OPERATOR:operator, VALUE:value, LOGIC:logic}

def checkFiltersValidity(filters):
   str = ''
   for i, filter in enumerate(filters):
      if (filter[DATA] is None) | (filter[OPERATOR] == '') | (filter[VALUE] is None):
         str += f"Filter #{i+1}, "
   return str

def createQueryFromFilters(filters):
   query = ''
   for filter in filters:
      data = filter[DATA]
      operator = filter[OPERATOR]
      value = filter[VALUE]

      if (data is None) | (operator is None) | (value is None):
         query = ''
         break

      if filter[LOGIC] is not None:
         query += ' & ' if filter[LOGIC] == AND else ' | '
     
      if isinstance(value, float):
         query += f"{data} {operator} {str(value)}"
      else:
         if (operator == CONTAINS) | (operator == STARTS_WITH) | (operator == ENDS_WITH) :
            query += f"{data}.str.{operator}('{value}') "
         else:
            query += f"{data} {operator} '{value}'"

   return query

def makeOperation(df, operation, values, num_added, num_added_before=0):
   if len(values) > 0:
      if num_added_before != 0:
            if (operation == SOUSTRACTION): 
               result = num_added_before - df[values[0]]
            elif (operation == DIVISION): 
               result = num_added_before / df[values[0]]
      else:
         result = df[values[0]]

      if operation == ADDITION:
         for val in values[1:]:
            result += df[val]
         result += num_added

      elif operation == MULTIPLICATION:
         for val in values[1:]:
            result *= df[val]
         if num_added != 0:
            result *= num_added

      elif (operation == SOUSTRACTION) | (operation == DIVISION):
         
         if (operation == SOUSTRACTION): 
            for val in values[1:]:
               result -= df[val]
            if num_added != 0:
               result -= num_added

         if operation == DIVISION:
            for val in values[1:]:
               result /= df[val]
            if num_added != 0:
               result /= num_added
      
   
      return result

def filterDataChart(df, col, val, df_chart):
   if df[val].dtype.kind in 'iuf':
      min_val = min(df[val])
      max_val = max(df[val])
      f = col.slider(val, min_val, max_val,(min_val, max_val))
      df_chart = df_chart[df_chart[val].between(*f)]
   else: 
      f = col.multiselect(val, pd.unique(df_chart[val]))
      if len(f) > 0:
         df_chart = df_chart.query(f"{val} == @f")
   return df_chart

def setMLInput(cols):
   input = {}
   col1, col2, col3 = st.columns([1,1,1])
   id = 0
   for c in cols:
      id += 1
      if id == 1:
         col = col1
      elif id == 2: 
         col = col2
      else: 
         col = col3
         id = 0

      if df[c].dtype.kind in 'iuf':
         input[c] = col.number_input(c, None)
      else:
         input[c] = col.text_input(c, None)
      
   return input

uploaded_file = None
dataset = None
df = pd.DataFrame()

with st.sidebar.header("DataVision"):
   data_import = st.radio("Import your dataset", ["Build-in", 'Upload'])

   if data_import == 'Build-in':
      dataset = st.sidebar.selectbox('Choose a build-in dataset', datasets)
   else:
      uploaded_file = st.sidebar.file_uploader('Choose your file')

st.header("Data Vision App")


if uploaded_file is not None:
   df = load_csv(uploaded_file)
   filename = f'{uploaded_file.name}_new'
elif dataset is not None:
   df = load_dataset(dataset)


if not df.empty:
   df_filtered = pd.DataFrame()

   saving_data = "Saving Data"
   if ('df_modified' in  st.session_state) | ('model' in  st.session_state):
         saving_data += ' •'
   tab1, tab2, tab3, tab4 = st.tabs([ 
      "Data Manipulation", "Data Visualization", "Machine Learning", saving_data])
   
   ####################################################
   # DATA OVERVIEW ###################################
   with tab2:
      st.write("")

      if "df_original" not in st.session_state:
         st.session_state["df_original"] = df

      if 'df_modified' in st.session_state:
         df = st.session_state['df_modified']
      
      st.dataframe(df, height=250)


      ####################################################
      # NA Value #########################################
      with st.expander("Profile Report"):
         if st.button('Get Profile Report !'):
            pr = ProfileReport(df, explorative = True)
            st_profile_report(pr)

      with st.expander("Handling null values"):
            is_changes_applied = False

            if  'df_modified' in st.session_state:
               df_changed = st.session_state['df_modified']
               fig_na = visualizeNaValues(df_changed)
               is_changes_applied = True
            else:
               fig_na = visualizeNaValues(df)

            column = None
            col1, col2, col3 = st.columns([1,1,1])
            na_of = col1.selectbox("For", ['All the Data', 'Specific Column'])

            if na_of == 'All the Data':
               na_select = col2.selectbox("NA", na_state)
               if na_select == REPLACE:
                  na_select = col3.text_input("➡️ Replace by")
            
            else:
               column = col2.selectbox("Columns", df.columns)
               col1, col2, col3 = st.columns([1,1,1])
               
               na_select = col1.selectbox("NA", na_state)
              
               if na_select == REPLACE:
                  na_select = col2.selectbox("➡️ Replace by", na_replacement)
                  if na_select == FILL:
                     if df[column].dtype.kind in 'iuf':
                        na_select = col3.number_input('➡️ Number')
                     else:
                        na_select = col3.text_input("➡️ Text")
            
               
            col1, col2, col3 = st.columns([1,2,1])

            if col1.button("Apply Changes"):
               df_na_changed = filterNaValues(df, column, na_select)
               fig_na = visualizeNaValues(df_na_changed)
               st.session_state["df_modified"] = df_na_changed
               is_changes_applied = True
               st.experimental_rerun()


            if is_changes_applied == True:
               if col3.button("❌ Remove"):
                  del st.session_state["df_modified"]
                  fig_na = visualizeNaValues(df)
                  st.experimental_rerun()

            st.plotly_chart(fig_na)

      with st.container():
         st.markdown("<h5 class='title_section'>Data Manipulation</h5>", unsafe_allow_html=True)
         df_modified = df.copy()
         cols = df.columns.to_list()
         cols_num = df.select_dtypes(include='number').columns.to_list()
         col_renamed = ''
         col_to_rename = ''
         col_to_subset = None
         col_to_drop = None
         operation_chosen = ''
         is_renamed = False
         result = None

         col1, col2 = st.columns([1, 1])
         action = col1.selectbox('Action', actions,
               help= """
               The results of the operations will be added in a new column
               in your dataframe, named by the operation you have chosen
               """)

      # Rename #################################################
      if action == COLUMNS:
         action_on_column = col2.selectbox('Action on Column', action_columns)

         if action_on_column == RENAME_COL:
            col_to_rename = col1.selectbox('Column', cols)
            col_renamed = col2.text_input('To')

            # Prevent to have duplicate columns name 
            if col_renamed not in cols:
               is_renamed = True
         
         elif action_on_column == DROP_COL:
            col_to_drop = col2.multiselect('Columns to Drop', cols)

         elif action_on_column == SUBSET:
            col_to_subset = col2.multiselect('Columns to Subset', cols)

      # Operation ################################################
      elif action == OPERATION:
         operation = col2.selectbox('Operation', action_operations)
         values = col1.multiselect(
               'Apply to Column(s)', cols_num,
               help="""
                  It will apply the operator on all columns selected.
                  For example, if you choose the "+" operator and select 3 columns,
                  then the result will be the addition of the 3 columns.
               """)
         
         if (operation == ADDITION) | (operation == MULTIPLICATION):
            num_added = col2.number_input('Add Number *(Optional)*')
            result = makeOperation(df, operation, values, num_added)

         elif (operation == SOUSTRACTION) | (operation == DIVISION):
            col1, col2 = st.columns([1,1])
            num_added_before = col1.number_input('Number before columns *(Optional)*')
            num_added = col2.number_input('Number after columns *(Optional)*')     
            result = makeOperation(df, operation, values, num_added, num_added_before)
         
         operation_chosen = operation

      # Percentage ###############################################
      elif action == PERCENTAGE:
            col1, col2 = st.columns([1,1])
            value1 = col1.selectbox("Value 1", cols_num)
            value2 = col2.selectbox("Value 2", cols_num)
            result = (df[value1] / df[value2]) * 100
      
      # Mean ####################################################
      elif action == MEAN:
         values = col2.multiselect("Values", cols_num)
         if values:
            result = 0
            for val in values:  
               result += df[val]
            result = result / len(values)
         
      if result is not None:
         operation = f'({operation_chosen})' if len(operation_chosen)>0 else ''
         df_modified[f'{action} {operation}'] = result


      ###########################################################
      # Apply Changes ###########################################
      
      col1, col2 = st.columns([.5, 2])
      my_msg = col2.empty()

      if col1.button("Apply !"):
         is_valid = True

         # If column renamed
         if (len(col_renamed) > 0) & (is_renamed == False):
            my_msg.warning(f'''Another columns has the name {col_renamed}''')
            is_valid = False
         else:
            df_modified.rename(columns={col_to_rename: col_renamed}, inplace = True)

         # If column drop
         if col_to_drop is not None:
            df_modified.drop(col_to_drop, axis=1, inplace = True)

         # If columns subset
         if col_to_subset is not None:
            df_modified = df[col_to_subset]

         if is_valid:
            st.session_state['df_modified'] = df_modified
            my_msg.info("Changes applied with success !")
            time.sleep(1.5)
            st.experimental_rerun()
               
   #####################################################
   # DATA VISUALIZATION ################################
   with tab1:     
      st.write("")
      with st.container():
         st.write("1- Select a chart / plot type:")
         col1, col2 = st.columns([1,1])
         chart_type = col1.selectbox('Types',chart_types)

         if chart_type == GEO:
            geo_type = col2.selectbox('Geo plot types', geo_types)


      with st.container():
         st.write("2- Choose your values")
         if chart_type == GEO: 
            params = populateParams(geo_type, df) 
         else:
            params = populateParams(chart_type, df)

      with st.expander("3- Filtering data (Optional):"):       
         filters = []
         is_filter_viable = True
         num_filters = 0
         is_filter_added = False

         st.write("")

         if "num_filters" not in st.session_state:
            st.session_state["num_filters"] = 0

         col1, col2, col3 = st.columns([1,2,1])
         if col1.button("Add filter"):
            st.session_state["num_filters"] += 1
            is_filter_added = True

         if st.session_state["num_filters"] > 0:
            if col3.button("Remove Filters"):
               st.session_state["num_filters"] = 0

         populateFiltersView(df, operators, logics, createFilters, filters)

      # To save is fig is created for preventing reloading when filtering the
      if 'is_fig' in st.session_state:
            is_fig = st.session_state["is_fig"]
      else:
         is_fig = False

      if st.button('Create Chart!'):
         filters_no_valid = ''
         is_fig = True

         # Check Validity of filters
         if filters:
            filters_no_valid = checkFiltersValidity(filters)
            if filters_no_valid:
               is_fig = False
               st.warning(filters_no_valid + " contain(s) None or empty values")
            else:         
               my_query = createQueryFromFilters(filters)
               df_filtered = df.query(my_query, engine='python')

         if "df_na_changed" in st.session_state:
            df_filtered = st.session_state["df_na_changed"]

         if 'is_fig' not in st.session_state:
            st.session_state["is_fig"] = is_fig

      if is_fig:
         df_chart = df if df_filtered.empty else df_filtered
      
         # Filters #######################################
         # with st.expander("Figure Filters"):
         #    col1, col2 = st.columns([1, 1], gap='medium')
            
         #    # X axis
         #    x = params['x']
         #    if chart_type == COUNT_PLOT:
         #       df[x] = pd.Categorical(df[x])     
         #    df_chart = filterDataChart(df, col1, x, df_chart) 

            # Color group
            # if "color" in params:
            #    color = params['color']

            #    if chart_type == COUNT_PLOT:
            #       df[x] = pd.Categorical(df[x])
            #       df[color] = pd.Categorical(df[color])

            #    if "color_type" in params:
            #       df[color] = pd.Categorical(df[color])
               
            #    df_chart = filterDataChart(df, col2, color, df_chart) 
     
         if chart_type == GEO:
            fig = geo.createGeoPlot(geo_type, df_chart, params)
            if geo_type == SCATTER_MAP:
               st.map(fig, use_container_width=True)
            elif geo_type == PYDECK_CHART:
               st.pydeck_chart(fig)
         else:
            fig = ch.createFigure(chart_type, df_chart, params)
            st.plotly_chart(fig)


   #####################################################
   # MACHINE LEARNING ##################################
   with tab3:
      from sklearn.linear_model import LogisticRegression
      from sklearn.preprocessing import StandardScaler
      from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
      from sklearn import metrics

      color_palette = px.colors.qualitative.Vivid

      st.write("")

      ml_action = st.radio('Action',
               ["Build a new Model","Upload a Model"], horizontal=True)

      if ml_action == "Upload a Model":
         with st.container():
            st.markdown("<h5 class='title_section'>Upload a Model </h5><br>", unsafe_allow_html=True)
            col1, col2 = st.columns([1,.3])
            uploaded_model = col1.file_uploader('Choose your Model', )

      else:
         with st.container():
            st.markdown("<h5 class='title_section'>Build a new Model </h5><br>", unsafe_allow_html=True)

            df_ml = df.copy()
            cols = df_ml.columns
            is_build_model = False
            ml_params = {}
            my_input = None

            col1, col2, col3 = st.columns([1,1,.5])
            model_wanted = col1.selectbox('Models', ml_models)
            
            col1, col2, col3 = st.columns([1, 1, .5])
            train_test_values = col1.multiselect('Values to Train / Test the model',
                           cols, [col for col in cols])
            target = col2.selectbox('Target Value', cols)
            val_to_scale = col2.multiselect('Values to scale (Optional)', cols)
            test_size = col3.number_input('Test Size', min_value=0.0, max_value=1.0, value=.2,
               help = """
                  A commonly used ratio is 80:20, which means 80 perc of 
                  the data is for training and 20 prec for testing, i.e Test Size = 0.2. 
               """)
            radio_input = st.radio("Add your own input ?", ['No', 'Yes'], horizontal=True)
            
            if model_wanted == KNN:
               n_neighbors = col3.number_input('N Keighbors', min_value=1, value=1)
               ml_params['n_neighbors'] = n_neighbors

            elif model_wanted == RANDOM_FOREST:
               n_estimators = col3.number_input('N Estimators', min_value=1, value=100)
               ml_params['n_estimators'] = n_estimators

            if radio_input == 'Yes':
               with st.expander('Your input'):
                  my_input = setMLInput(df_ml.drop(target, axis=1).columns)
            
            if st.button("Build Model !"):
               if my_input:
                  for val in my_input.values():
                     if val == "None":
                        st.warning("Your input contains None values")
                        break
               else:
                  is_build_model = True

         ###################################################################
         # Results #########################################################
         if is_build_model:
            is_my_input = True if my_input else False

            st.write('')
            st.markdown("<h5 class='title_section'>Results </h5>", unsafe_allow_html=True)

            with st.expander('1- Exploratory Data Analysis'):
               with st.spinner('Analyzing the data...'):
                  st.write('')
                  
                  ml.createDataExploration(df_ml, model_wanted, target)
            
            with st.expander(f"2- Report for {model_wanted}"):
               with st.spinner('Creating ML Model...'):
                  st.write('')
      
                  if len(val_to_scale) > 0:
                     StandardScaler = StandardScaler()
                     df_ml[val_to_scale] = StandardScaler.fit_transform(df_ml[val_to_scale])

                  X = df_ml[train_test_values]
                  y = df_ml[target]
                  X_train, X_test, y_train, y_test = ml.trainAndTestSplit(df_ml, train_test_values, target, test_size)
                  
                  # SVM needs Grid search model then I don't place it with the others model
                  if model_wanted == SVM:
                     from sklearn.model_selection import GridSearchCV
                     from sklearn.svm import SVC

                     param_grid = {
                        'C': [0.1,1, 10, 100, 1000], 
                        'gamma': [1,0.1,0.01,0.001,0.0001], 
                        'kernel': ['rbf']}
                  
                     grid = GridSearchCV(SVC(),
                           param_grid, refit=True, verbose=3)

                     with st.spinner("May take a while..."):
                        grid.fit(X_train, y_train)
                        st.write("Best Params : ",grid.best_params_)
                        st.write("Best Estimators : ", grid.best_estimator_)
                        st.write('')
                        predictions = grid.predict(X_test)

                        st.markdown("<h5 class='title_section'>Classification Report</h5>", unsafe_allow_html=True)
                        cr = classification_report(y_test, predictions, output_dict=True)
                        st.dataframe(pd.DataFrame(cr).transpose())
                  else:
                     model = ml.returnMLModel(model_wanted, ml_params)
                     model.fit(X_train, y_train)
                     predictions = model.predict(my_input if is_my_input else X_test)
                  
                  if model_wanted == LINEAR:
                     st.markdown("<h5 class='title_section'>Coefficient</h5>", unsafe_allow_html=True)
                     coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
                     st.dataframe(coeff_df)

                     df_ml['predictions'] = model.predict(X)

                     st.markdown("<h5 class='title_section'>Enhanced prediction error</h5>", unsafe_allow_html=True)
                     fig = px.scatter(
                        df_ml, x=target, y='predictions',
                        marginal_x='histogram', marginal_y='histogram',
                        color='split', trendline='ols',
                        color_discrete_sequence=color_palette,
                     )
                     fig.update_traces(histnorm='probability', selector={'type':'histogram'})
                     fig.add_shape(
                        type="line", line=dict(dash='dash'),
                        x0=df_ml[target].min(), y0=df_ml[target].min(),
                        x1=df_ml[target].max(), y1=df_ml[target].max()
                     )
                     st.plotly_chart(fig, use_container_width=True)

                     st.markdown("<h5 class='title_section'>Residual plots</h5>", unsafe_allow_html=True)
                     df_ml['residual'] = df_ml['predictions'] - df_ml[target]

                     fig = px.scatter(df_ml, x='predictions', y='residual',
                        marginal_y='violin',
                        color='split', trendline='ols',
                        color_discrete_sequence=color_palette,

                     )
                     st.plotly_chart(fig, use_container_width=True)
                  
                     st.markdown(f"<h5 class='title_section'>Metrics</h5>", unsafe_allow_html=True)
                     st.write('MAE:', metrics.mean_absolute_error(y_test, predictions))
                     st.write('MSE:', metrics.mean_squared_error(y_test, predictions))
                     st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
                     st.write(
                        """
                           - **MAE** is the average error.
                           - **MSE** is more popular than MAE, because MSE "punishes" larger errors, useful in the real world.
                           - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
                           All of these are **loss functions**, because we want to minimize them.
                        """
                     )
               
                  elif (model_wanted == LOGISTIC) | (model_wanted == KNN) | \
                     (model_wanted == RANDOM_FOREST) | (model_wanted == DECISION_TREE):
                        
                     st.markdown("<h5 class='title_section'>Confusion Matrix</h5>", unsafe_allow_html=True)
                     cm = confusion_matrix(y_test, predictions)
                     st.plotly_chart(px.imshow(cm, text_auto=True), use_container_width=True)

                     st.markdown("<h5 class='title_section'>Classification Report</h5>", unsafe_allow_html=True)
                     cr = classification_report(y_test, predictions, output_dict=True)
                     st.dataframe(pd.DataFrame(cr).transpose())
                     
                     accuracy = accuracy_score(y_test, predictions)

                      
                     st.write(f"On {'your input ' if is_my_input else 'testing'} data :")
                     st.write("Accuracy : ", accuracy)
                  
                 
            if model_wanted == KNN:
                  with st.expander("3- Find the best K Value (Elbow Method)"):
                     with st.spinner('One moment please...'):
                        st.write('')
                        error_rate = []
                        for i in range(1,40):
                           ml_params['n_neighbors'] = i
                           model = ml.returnMLModel(model_wanted, ml_params)
                           model.fit(X_train, y_train)
                           pred_i = model.predict(X_test)
                           error_rate.append(np.mean(pred_i != y_test))
                        
                        fig=plt.figure(figsize=(10,6))
                        plt.plot(range(1,40),
                                 error_rate,
                                 color='dodgerblue',
                                 linestyle='dashed', 
                                 marker='o',
                                 markerfacecolor='red', 
                                 markersize=8)
                        plt.title('Error Rate vs. K Value')
                        plt.xlabel('K')
                        plt.ylabel('Error Rate')

                        st.pyplot(fig)
   
            # if "model" not in st.session_state:
            #    st.session_state['model'] = model
            # else:
            #    st.session_state['model'] = model

   #####################################################
   # SAVING RESULT #####################################
   with tab4:
      with st.container():
         col1, col2 = st.columns([1,1])
                     
         if not df.empty:
            csv = convert_df(df)
            col1.markdown("<h5 class='title_section'>Dataset</h5>", unsafe_allow_html=True)
            filename_csv = col1.text_input("File Name", filename)
            col1.download_button(
               label="Download Dataset",
               data=csv,
               file_name=f'{filename_csv}.csv',
               mime='text/csv')
         
         # col2.markdown("<h5 class='title_section'>Machine Learning Model</h5>", unsafe_allow_html=True)
         # filename_csv = col2.text_input("File Name", 'ml_model')
         # col2.download_button(
         #       label="Save Model",
         #       data=pickle.dumps(model),
         #       file_name=f'{model_wanted}_model')
       
else:
   st.info("Awaiting for data to be uploaded")
  
