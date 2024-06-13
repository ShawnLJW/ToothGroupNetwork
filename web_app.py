import base64
import io
import numpy as np
import plotly.graph_objects as go
import gen_utils as gu
import open3d as o3d
from dash import Dash, html, dcc, callback, Output, Input, State
import trimesh
from inference_pipelines.inference_pipeline_maker import make_inference_pipeline

def predict_teeth(mesh):
    return np.random.randint(0, 9, len(mesh.vertices))

def get_plotly_mesh(mesh):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color((0.9, 0.9, 0.87))
    vertexcolor = np.asarray(mesh.vertex_colors)
    mesh_3d = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        vertexcolor=vertexcolor,
        flatshading=True,
        lighting=dict(
            ambient=0.18,
            diffuse=1,
            fresnel=0.1,
            specular=1,
            roughness=0.05,
            facenormalsepsilon=1e-15,
            vertexnormalsepsilon=1e-15,
        ),
        lightposition=dict(x=100, y=200, z=0),
    )
    fig = go.Figure(data=[mesh_3d])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )
    return fig

@callback(
    Output('mesh-container', 'children'),
    Input('upload-button', 'contents'),
    State('upload-button', 'filename'),
)
def upload_mesh(contents, filename):
    global mesh
    if contents is None:
        return html.Div(children='Upload a 3D mesh to visualise it', id="mesh-container", style={'textAlign':'center'})
    content_type, content_string = contents.split(',')
    mesh_path = io.BytesIO(base64.b64decode(content_string))
    trimesh_mesh = trimesh.load(mesh_path, file_type=filename.split('.')[-1])
    vertex_ls = np.array(trimesh_mesh.vertices)
    tri_ls = np.array(trimesh_mesh.faces)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(tri_ls)
    mesh.compute_vertex_normals()
    return dcc.Graph(figure=get_plotly_mesh(mesh), id="mesh-object")

@callback(
    Output('mesh-object', 'figure'),
    Input("label-button", "n_clicks"),
)
def label_teeth(n_clicks):
    global mesh
    if mesh is not None:
        print("Labelling teeth")
        outputs = pipeline(mesh)
        lables = np.array(outputs["sem"])
        mesh = gu.get_colored_mesh(mesh, lables)
        return get_plotly_mesh(mesh)

pipeline = make_inference_pipeline(
    model_name="tgnet",
    ckpt_path_ls=["ckpts/tgnet_fps.h5", "ckpts/tgnet_bdl.h5"],
)

mesh = None
app = Dash(__name__)
app.layout = [
    html.H1(children='Teeth Segmentation', style={'textAlign':'center'}),
    html.Button("Clear", id="clear-button", style={'margin':'auto', 'display':'block'}),
    dcc.Upload(
        id='upload-button',
        children=html.Button('Upload Data', style={'margin':'auto', 'display':'block'}),
    ),
    html.Div(children='Upload a 3D mesh to visualise it', id="mesh-container", style={'textAlign':'center'}),
    html.Button("Label Teeth", id="label-button", style={'margin':'auto', 'display':'block'}),
]

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)