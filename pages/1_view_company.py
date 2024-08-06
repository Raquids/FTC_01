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


st.set_page_config( page_title='Company View', page_icon = "🥜", layout = 'wide') 

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
#            /Function
# ==================================

# importing dataset
df = pd.read_csv('datasets\\food-delivery-dataset\\train.csv')

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
    
    st.sidebar.markdown( '\n ' )
    
    
    
    traffic_tags = st.sidebar.multiselect('Which were the Traffic condiditons?', 
                                          list(df1['Road_traffic_density'].unique()),
                                          default = list(df1['Road_traffic_density'].unique() ) )
    
    st.sidebar.markdown( '''---------''' )
    st.sidebar.markdown( 'Powered by: the Shadow Government ' )

# Making the sidebar filter work
selected_lines = (df1['Order_Date'] <= date_slider) & (df1['Road_traffic_density'].isin(traffic_tags))
df1 = df1.loc[selected_lines, :] 

st.dataframe( df1 )
# ===================================
#           Tabs
# ==================================

tab1,tab2,tab3 = st.tabs( ["Manager View","Tactical View","Geo View"] )


# Manager View
# ---------------------------------------------
with tab1:
    with st.container():
        st.markdown( "# Orders by Day" )
        # selected visualization of the dataframe
        df_aux = df1[['ID','Order_Date']].groupby('Order_Date').count().reset_index()
         
        fig1 = px.bar(x=df_aux['Order_Date'],y=df_aux['ID'])
        st.plotly_chart(fig1, use_container_width = True) 

    with st.container():
        col1,col2 = st.columns(2)
        with col1:
            st.markdown( "# Traffic Order Share" )
            df_aux = df1.loc[:,['Road_traffic_density','ID']].groupby(['Road_traffic_density']).count().reset_index()
            df_aux['percentile'] = 100 * (df_aux['ID']/df_aux['ID'].sum())

            fig2 = px.pie(df_aux,values='percentile',names = 'Road_traffic_density')
            st.plotly_chart(fig2, use_container_width = True) 
            
        with col2:
            st.markdown( "# Traffic Order City" )
            df_aux1 = df1.loc[:,['ID','Road_traffic_density','City']].groupby(['City','Road_traffic_density']).count().reset_index()

            fig3 = px.scatter(df_aux1, x = 'City', y = 'Road_traffic_density', color = 'City', size = 'ID')
            st.plotly_chart(fig3, use_container_width = True) 



# Tactical View
# ---------------------------------------------
with tab2:
    with st.container():
        st.markdown( "Order by Week" )
        df_aux = df1.loc[:,['Delivery_person_Ratings','Week']].groupby(['Week']).mean().reset_index()
    
        fig4 = px.bar(df_aux,x='Delivery_person_Ratings',y='Week')
        st.plotly_chart(fig4, use_container_width = True) 
    with st.container():

        # selected visualization of the dataframe
        df_aux1 = df1[['ID','Week']].groupby('Week').count().reset_index()
        df_aux2 = df1[['Delivery_person_ID','Week']].groupby('Week').nunique().reset_index()
        
        df_aux = pd.merge( df_aux1, df_aux2, how= 'inner', on='Week')
        df_aux['Order_by_deliver'] = df_aux['ID']/df_aux['Delivery_person_ID']
        
        #but how to plot_this_shit.png
        fig5 = px.line(x=df_aux['Week'],y=df_aux['Order_by_deliver'])
        
        st.plotly_chart(fig5, use_container_width = True) 



# Geo View
# ---------------------------------------------
with tab3:
    st.markdown( "Country Maps" )
    df_aux = df1.loc[:,['Road_traffic_density','Delivery_latitude','Delivery_longitude','City']].groupby(['City','Road_traffic_density']).median().reset_index().dropna()
    # mapping the mappydy doo
    map_ = folium.Map(zoom_start=18)

    for index, location_info in df_aux.iterrows():
        popup_content = f"City: {location_info['City']}<br>Traffic Density: {location_info['Road_traffic_density']}"
        folium.Marker([location_info['Delivery_latitude'],
                       location_info['Delivery_longitude']],
                       popup = folium.Popup(popup_content, max_width=300)
                     ).add_to(map_)
    folium_static(map_, width=1024, height=600)



