import streamlit as st
from PIL import Image

# Home page Setup
st.set_page_config(
    page_title = 'Page_title',
    page_icon = '🐑'
   )

# Sidebar Setup
# image_path = r'C:\Users\jgabr\Desktop\repos\cds\FTC\assets\indiani_banani.png'
image = Image.open('assets/indiani_banani.png')
st.sidebar.image(image, width=120)

st.sidebar.markdown( '# Cury Company' )
st.sidebar.markdown( '## Fastest Delivery in town') 
st.sidebar.markdown('''----------''' )

# Home page content

# Header
st.write( '# Curry company Growth Dashboard' ) 

# Body
st.markdown(
    '''
    Growth Dashboard developed to track KPIs and growth of #

    ## How to use:
    ### Company View:
        - Manager View:   General Behavioral metrics
        - Tactical View:  Weekly growth trackers 
        - Geo View:       Geographic insights 
    ### -Delivery View: 
        - track weekly KPIs and growth tendencies 


    ## Help me!
    - Contact @
 ''')
