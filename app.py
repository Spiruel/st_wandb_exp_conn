import streamlit as st
from st_files_connection import FilesConnection
import numpy as np

class WandbConnector:

    def __init__(self):
        self._conn = None

    def _connect(self):
        self._conn = st.experimental_connection('wandb', type=FilesConnection)

    def cursor(self):
        if not self._conn:
            self._connect()
        return self._conn

    def query(self, path, method='read', **kwargs):
        conn = self.cursor()
        
        if method == 'ls':
            return conn.fs.ls(path, **kwargs)
          
        elif method == 'open':
            return conn.open(path, **kwargs)
        
        elif method == 'read':
            return conn.read(path, **kwargs)
        else:
            raise ValueError(f"Invalid method: {method}")

@st.cache_data(ttl='1h')
def cached_query(_connector, path):
    return _connector.query(path)
    
st.title('st.experimental_connection with Weights & Biases')
st.info('''This Streamlit app demonstrates how to access and display data from Weights & Biases machine learning experiments using the new experimental_connection functionality. 

The app establishes a connection to W&B cloud storage using the FilesConnection object from st_files_connection. This allows browsing W&B entities, projects, and files.

This app is just a basic demo for now, but can be used to create custom interfaces to your model runs.''')

wandb = WandbConnector() 

entity = st.text_input('Wandb Entity eg. `wandb`, `sciml-leeds`', value='wandb')

col1, col2 = st.columns([0.7,0.3])
a = col1.empty()
b = col2.empty()

try:
    projects =  wandb.query(entity, method='ls')
    projects = list(set(projects))
except Exception as e:
    st.error(f'Could not find projects under {entity}')
else:    
    random_project = col1.button('Select random project')
    
    if random_project:
        project = np.random.choice(projects)
        project_index = projects.index(project)
    else:
        project_index = 0
        
    try:
        project = a.selectbox('Select project', projects, index=project_index) 
        project = project.split('/')[-1]
    except Exception as e:
        st.error(f'Could not find projects under {entity}')
    else:

        runs = wandb.query(f'{entity}/{project}', method='ls')
        if len(runs) == 0:
            st.warning('No runs found')
        else:
            runs = [r.split('/')[-1] for r in runs]
        
            random_run = col2.button('Select random run')
        
            if random_run:
                run_id = np.random.choice(runs)
                run_index = runs.index(run_id)
            else:
                run_index = 0
                
            run_id = b.selectbox('Select run', runs, index=run_index)

            files = wandb.query(f'{entity}/{project}/{run_id}', method='ls')
        
            graph_check = ['graph' in f for f in files]
            if any(graph_check):
                with st.spinner('loading...'):
                    st.markdown('#### graph')
                    graph = wandb.query(f'{entity}/{project}/{run_id}/graph', method='read')
                    st.graphviz_chart(graph, use_container_width=True)
               
            media_check = ['media' in f for f in files]
            if any(media_check):
                with st.spinner('loading images...'):
                    media_src = np.array(files)[media_check]
                    for m_s in media_src:
                        images = wandb.query(m_s+'/images', method='ls')
                        st.markdown('#### images')
                        if len(images) == 0:
                            st.warning('no images')
                            
                        images = [i for i in images if '.png' in i][:21]
        
                        n = len(images)
                        num_row = 7
                        rows = n//num_row
        
                        images_conn = [wandb.query(img, method='open') for img in images]
        
                        for row_idx in range(0, rows, 1):
                            st.image(images_conn[0+row_idx*num_row:0+row_idx*num_row+num_row], width=100)
                                
            if len(files)>0:
                st.markdown('#### files')
                st.table(files)
                
            reqs_check = ['requirements.txt' in f for f in files]
            if any(reqs_check):
                with st.expander('run requirements:'):
                    reqs = cached_query(wandb, f'{entity}/{project}/{run_id}/requirements.txt')
                    st.code(reqs)


