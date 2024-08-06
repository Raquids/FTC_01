# ===================================
#            Libs                  
# ==================================
         
   
import re                               # that regex stuff         
import folium                           # folium to plot mappy maps
import matplotlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px 
import plotly.graph_objects as go

from PIL import Image
from datetime import datetime
from haversine import haversine
from streamlit_folium import folium_static


st.set_page_config( page_title='Delivery View', page_icon = '⛏', layout = 'wide')

# ===================================
#            Functions
# ==================================

def distance_calculator (df1):
    # utilizing haversine lib to calculate and storing in a new column
    coordinates = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_latitude', 'Delivery_longitude']
    
    df1['Distance'] = df1.loc[:,coordinates].apply( lambda x: haversine( ( x[ coordinates[0] ], x[ coordinates[1] ]),
                                                                         ( x[ coordinates[2] ], x[ coordinates[3] ])   ),   axis = 1)
    return df1


def pre_processing (df1):

            # 1.Removing backspace from the dataframe
    df1 = df1.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
                # 4.1 removing "(min) " string so we are able to convert into a numerical object with regex.
    df1['Time_taken(min)'] = df1['Time_taken(min)'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]))
    
                #4.2 removing "conditions " from each cell for better visualization 
    df1['Weatherconditions'] = df1['Weatherconditions'].str.replace("conditions ", '')
    
        # 2.replacing for np.nan so we can fill easier
    df1 = df1.replace(['NaN', 'NaN '], np.nan)
    
        # 3. supress scientific notation
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
        # 4. String correction within columns
    
                # 4.1.1 handling exceptions so the type transformation will process np.nans
    #df1['Time_taken(min)'] = df1['Time_taken(min)'].apply(lambda x: int(x[0]) if x else None)
    
    
    
        # 5.Renaming columns
    df1 = df1.rename(columns={
        'Delivery_location_latitude': 'Delivery_latitude',
        'Delivery_location_longitude': 'Delivery_longitude'
    })
    
    
    # 6. Selecting which columns we desire to disconsider np.nan rows. 
    
    selected_rows = (
        (df1['Delivery_person_Age'].notna())  &
        (df1['multiple_deliveries'].notna())  &
        (df1['City'].notna())                 &
        (df1['Road_traffic_density'].notna()) &
        (df1['Festival'].notna())             &
        (df1['Delivery_latitude'].notna())    &
        (df1['Delivery_longitude'].notna())   &
        ((df1['Delivery_person_Ratings'].notna()))
    )
        #6.1 select to me only the lines, within all columns, that attend to query == True
    df1 = df1.loc[selected_rows,:].copy() 
    
    
    #7. Type Convesions
        # 7.1 to int
    selected_cols = ['Delivery_person_Age','multiple_deliveries']
    for col in selected_cols:
        df1[col] = df1[col].astype(int)
    
        # 7.2 to float
    df1['Delivery_person_Ratings'] = df1['Delivery_person_Ratings'].astype(float)
     
       #7.3 to category
    category_columns = (['Road_traffic_density'            ,
                         'multiple_deliveries'             ,
                         'Vehicle_condition'               ,
                         'Weatherconditions'               ,
                         'Type_of_vehicle'                 ,
                         'Type_of_order'                   ,
                         'Festival'                        ,
                         'City'               ])
                        
    df1[category_columns] = df1[category_columns].astype('category')
    
    
        #7.4 datetime
    df1['Order_Date'] = pd.to_datetime(df1['Order_Date'], format='%d-%m-%Y')

    
    #8. Creating new columns
    
        # 8.1 Creating a column containing the distance between two points (latitude, longitude)
    distance_calculator(df1)
    

        # 8.2 Creating the columns Week to represent week number of the year
    df1['Week'] = df1['Order_Date'].dt.strftime('%U')

    return df1

# ===================================
#            Functions
# ==================================

# importing dataset
df = pd.read_csv('datasets/food-delivery-dataset/train.csv')

df1 = df.copy()

df1 = pre_processing(df1)

# ===================================
#            Sidebar
# ==================================

# niching all sidebars together
with st.sidebar.container():
    image = Image.open('assets/indiani_banani.png')
    st.sidebar.image(image, use_column_width=True)
    
        
    st.sidebar.markdown( '# Cury Company' )
    st.sidebar.markdown( '### Fastest Delivery in Town' )
    st.sidebar.markdown( '''---------''' )
    st.sidebar.markdown( '## Select the date to limit your view:' )
    
    # adds a slider to the side-bar where user can input date
    date_slider = st.sidebar.slider( 'Up until ',
                                    value =     datetime(2022, 4, 13),
                                    min_value = datetime(2022, 2, 11),
                                    max_value = datetime(2022, 4, 6), 
                                    format='DD-MM-YYYY' )

    date_slider_datetime = pd.to_datetime(date_slider, format='%d-%m-%Y')  # Corrected the date format
    st.sidebar.markdown('\n ')
    
    # adds a sidebar traffic conditions filter
    traffic_tags = st.sidebar.multiselect('Select Traffic Condiditons:', 
                                          list(df1['Road_traffic_density'].unique()),
                                          default = list(df1['Road_traffic_density'].unique() ) )

    
    # adds a sidebar Weather conditions filter
    weather_tags = st.sidebar.multiselect('Select Weather Condiditons:', 
                                            list(df1['Weatherconditions'].unique()),
                                            default = list(df1['Weatherconditions'].unique() ) )
    # adds a sidebar City  filter
    city_tags = st.sidebar.multiselect('Select Cities:', 
                                            list(df1['City'].unique()),
                                            default = list(df1['City'].unique() ) )

    # Festival status filter, i dont like this one
    festival_tags = st.sidebar.multiselect("Festival Status", list(df1['Festival'].unique()), default=list(df1['Festival'].unique() )  )

    # Festival status filter, i dont like this one
    mdelivery_tags = st.sidebar.multiselect("Multiple Deliveries", list(df1['multiple_deliveries'].unique()), default=list(df1['multiple_deliveries'].unique() )  )

        
    st.sidebar.markdown( '''---------''' )
    st.sidebar.markdown( 'Powered by: the Shadow Government ' )

    # Ensure dates in DataFrame are datetime objects
df1['Order_Date'] = pd.to_datetime(df1['Order_Date'], format='%d-%m-%Y')

# Normalize the Order_Date column to remove time component
df1['Order_Date'] = df1['Order_Date'].dt.normalize()

selected_lines = ( (df1['Order_Date'] <= date_slider_datetime)  & 
    (df1['multiple_deliveries'].isin(mdelivery_tags))          & 
    (df1['Road_traffic_density'].isin(traffic_tags))          &
    (df1['Weatherconditions'].isin(weather_tags))            &
    (df1['Festival'].isin(festival_tags))                   &
    (df1['City'].isin(city_tags))     )
    

filtered_df = df1.loc[selected_lines, :]

# Display the filtered DataFrame
st.dataframe(filtered_df)
# ===================================
#           Tabs
# ==================================

# Define colorblind-friendly colors for each category
colors = px.colors.qualitative.Set1

# Manager View
# ---------------------------------------------

with st.container():
    col1,col2 = st.columns(2)
        
    with col1:

        # configuring the first graph 
        df_aux = df1[['Delivery_person_ID', 'Delivery_person_Age']].groupby('Delivery_person_Age').count().reset_index()
        df_aux.rename(columns={'Delivery_person_ID': 'Count'}, inplace=True)
        
        fig1 = px.bar(
            df_aux,
            x='Delivery_person_Age',
            y='Count',
            labels={'Delivery_person_Age': 'Age', 'Count': 'Occurrences'},
            color='Delivery_person_Age',
            title='Occurrences of Ages of Delivery Persons'
        )
     
        # title addition 
        fig1.update_layout(title_text='Drivers Age Distribution Chart')
        
        st.plotly_chart(fig1, use_container_width = True) 
       
    with col2: 
         # configuring the 2nd graph 
        df_aux2 = df1[['ID', 'Vehicle_condition']].groupby('Vehicle_condition').count().reset_index()
        fig2 = px.pie(df_aux2,values = 'ID', names='Vehicle_condition')
        # title addition 
        fig2.update_layout(title_text='Vehicle Condition Distribution Chart') # assuming each driver only has 1 vehicle
        
        st.plotly_chart(fig2, use_container_width = True) 
            
    
with st.container():
    col1,col2 = st.columns(2)

    with col1:
        st.subheader('Weather Condition CSAT')

        df_weather = df1.groupby('Weatherconditions')['Delivery_person_Ratings'].mean().reset_index()

        # Convert 'Delivery_person_Ratings' to numeric, coercing errors
        df_weather['Delivery_person_Ratings'] = pd.to_numeric(df_weather['Delivery_person_Ratings'], errors='coerce')
        
        # Handle NaN values, either fill with a default value or drop
        df_weather.dropna(subset=['Delivery_person_Ratings'], inplace=True)  # Dropping rows with NaN in 'Delivery_person_Ratings'
        
        fig_weather = px.bar(
            df_weather,
            y='Weatherconditions',
            x='Delivery_person_Ratings',
            orientation='h',
            labels={'Weatherconditions': 'Weather Condition', 'Delivery_person_Ratings': 'Average CSAT Rating'},
            title='Average CSAT Ratings by Weather Conditions',
            color='Delivery_person_Ratings'
        )
        st.plotly_chart(fig_weather, use_container_width=True)

    # Second column - Box Plot
    with col2:
        st.subheader('City CSAT')
        df_city = df1.groupby('City')['Delivery_person_Ratings'].mean().reset_index()
        fig_city = px.box(
            df1,
            x='City',
            y='Delivery_person_Ratings',
            labels={'City': 'City', 'Delivery_person_Ratings': 'CSAT Rating'},
            title='Distribution of CSAT Ratings by City'
        )
        st.plotly_chart(fig_city, use_container_width=True)

