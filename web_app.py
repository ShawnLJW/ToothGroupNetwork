import argparse
import base64
import io
import numpy as np
import plotly.graph_objects as go
import gen_utils as gu
import open3d as o3d
from dash import Dash, html, ctx, dcc, callback, Output, Input, State, MATCH, ALL
import trimesh
from inference_pipelines.inference_pipeline_maker import make_inference_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="tgnet", help="tgnet | tsegnet | pointtransformer | pointnetpp | pointnet | dgcnn")
parser.add_argument("--ckpt", default="ckpts/tgnet_fps.h5", help="Path to the checkpoint file")
parser.add_argument("--ckpt_bdl", default="ckpts/tgnet_bdl.h5", help="Path to the checkpoint file for tgnet BDL module")
args = parser.parse_args()

pipeline = make_inference_pipeline(
    model_name = args.model_name,
    ckpt_path_ls = [args.ckpt, args.ckpt_bdl],
)

FDI_NUMBERING = {
    0: "Jaw",
    11: "Upper Right Central Incisor",
    12: "Upper Right Lateral Incisor",
    13: "Upper Right Canine",
    14: "Upper Right First Premolar",
    15: "Upper Right Second Premolar",
    16: "Upper Right First Molar",
    17: "Upper Right Second Molar",
    18: "Upper Right Third Molar",
    21: "Upper Left Central Incisor",
    22: "Upper Left Lateral Incisor",
    23: "Upper Left Canine",
    24: "Upper Left First Premolar",
    25: "Upper Left Second Premolar",
    26: "Upper Left First Molar",
    27: "Upper Left Second Molar",
    28: "Upper Left Third Molar",
    31: "Lower Left Central Incisor",
    32: "Lower Left Lateral Incisor",
    33: "Lower Left Canine",
    34: "Lower Left First Premolar",
    35: "Lower Left Second Premolar",
    36: "Lower Left First Molar",
    37: "Lower Left Second Molar",
    38: "Lower Left Third Molar",
    41: "Lower Right Central Incisor",
    42: "Lower Right Lateral Incisor",
    43: "Lower Right Canine",
    44: "Lower Right First Premolar",
    45: "Lower Right Second Premolar",
    46: "Lower Right First Molar",
    47: "Lower Right Second Molar",
    48: "Lower Right Third Molar",
}
label_map = np.vectorize(FDI_NUMBERING.get)

def get_plotly_mesh(mesh, labels=None):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color((0.9, 0.9, 0.87))
    vertexcolor = np.asarray(mesh.vertex_colors)
    if labels is None:
        labels = ["Unlabelled"] * len(vertices)
    mesh_3d = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        text=labels,
        hovertemplate="%{text}<extra></extra>",
        vertexcolor=vertexcolor,
        flatshading=True,
        lighting=dict(
            ambient=0.25,
            diffuse=1,
            fresnel=0.1,
            specular=1,
            roughness=0.05,
            facenormalsepsilon=1e-15,
            vertexnormalsepsilon=1e-15,
        ),
        lightposition=dict(x=100, y=200, z=0),
    )
    fig = go.Figure(data=mesh_3d)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )
    return fig

@callback(
    Output("mesh-container", "children"),
    Input({"type": "clear-button", "index": ALL}, "n_clicks"),
    Input({"type": "upload-button", "index": ALL}, "contents"),
    State({"type": "upload-button", "index": ALL}, "filename"),
    prevent_initial_call = True
)
def update_graph(n_clicks, contents, filename):
    triggered_id = ctx.triggered_id
    if triggered_id["type"] == "upload-button":
        return read_mesh(contents[0], filename[0])
    else:
        return upload_box()
    

def read_mesh(contents, filename):
    content_type, content_string = contents.split(",")
    mesh_path = io.BytesIO(base64.b64decode(content_string))
    trimesh_mesh = trimesh.load(mesh_path, file_type=filename.split(".")[-1])
    vertex_ls = np.array(trimesh_mesh.vertices)
    tri_ls = np.array(trimesh_mesh.faces)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(tri_ls)
    graph = dcc.Graph(
        figure=get_plotly_mesh(mesh),
        id={"type": "mesh-object", "index": 0},
        style={"flex": 1}
    )
    clear_button = html.Button(
        "Clear",
        id={"type": "clear-button", "index": 0}
    )
    label_button = html.Button(
        "Label Teeth",
        id={"type": "label-button", "index": 0}
    ) 
    button_row = html.Div(
        [clear_button, label_button],
        style={
            "display":"flex",
            "justify-content":"center",
            "gap":"2rem",
            "flex": 0
        }
    )
    return [button_row, graph]
        

@callback(
    Output({"type": "mesh-object", "index": MATCH}, "figure"),
    Input({"type": "label-button", "index": MATCH}, "n_clicks"),
    State({"type": "mesh-object", "index": MATCH}, "figure"),
    prevent_initial_call=True
)
def label_teeth(n_clicks, figure):
    plotly_mesh = figure["data"][0]
    vertex_ls = np.column_stack([plotly_mesh["x"], plotly_mesh["y"], plotly_mesh["z"]])
    tri_ls = np.column_stack([plotly_mesh["i"], plotly_mesh["j"], plotly_mesh["k"]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(tri_ls)
    mesh.compute_vertex_normals()
    try:
        outputs = pipeline(mesh, pca=True)
        labels = np.array(outputs["sem"])
        mesh = gu.get_colored_mesh(mesh, labels)
        return get_plotly_mesh(mesh, label_map(labels))
    except Exception as e:
        print("Error in labeling teeth")
        print(e)
        return figure

def upload_box():
    return dcc.Upload(
        id={"type": "upload-button", "index": 0},
        children="Drag and Drop or Click to Upload",
        style={
            "display":"flex",
            "width":"80vh",
            "height":"80vh",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "justify-content":"center",
            "align-items":"center",
            "cursor": "pointer"
        },
    )

app = Dash(__name__)
server = app.server
app.layout = [
    html.H1(children="Teeth Segmentation", style={"textAlign":"center"}),
    html.Div(
        upload_box(),
        id="mesh-container",
        style={
            "display":"flex",
            "flex-direction":"column",
            "width":"80vh",
            "height":"80vh",
            "margin": "auto",
        }
    ),
]

if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)