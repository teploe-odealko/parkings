import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import dash_table
from app import app
from DataLoader import DataLoader
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
import cv2
from PIL import Image
from Model.nnAPI import nnAPI

mapbox_access_token = "pk.eyJ1IjoiYmVzdHdpc2hlczEyMyIsImEiOiJja2YwemgwdTcweXl3MnFvYzVtaDExNmlkIn0.yowUCKCjiCTFXWhe8nzQCA"
value_dropdown = ["id", "lon", "lat", "source"]
label_dropdown = ["Id камеры", "Долгота", "Широта", "Источник"]
dataloader = DataLoader()
nnapi = nnAPI()
cameras_table = dataloader.get_cameras_table()

# Вкладка для загрузки камер через файл
tab1_from_file = dbc.Card(
    dbc.CardBody(
        [
            dbc.Alert("Загружайте в формате .csv", color="primary"),
            html.Div(id='filenames'),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Перетащите или ',
                    html.A('Выберете файлы')
                ]),
                style={
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                },
                multiple=True
            )
        ]
    ),
    className="mt-3",
)

# Вкладка для загрузки камер через вручную
tab2_manually = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)

# Всплывающее окно для добавление новых камер
adding_camera_modal = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("Добавление камер"),
                dbc.ModalBody(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(tab1_from_file, label="Из файла"),
                                dbc.Tab(tab2_manually, label="Вручную"),
                            ]
                        )
                    ]

                ),
                dbc.ModalFooter(

                    dbc.Button(
                        "Добавить", id="modal_button", className="ml-auto"
                    )
                ),
            ],
            id="modal-centered",
            centered=True,
            style={"width": "40%", "max-width": "initial"}
        ),
    ]
)

cameras_content = html.Div(
    [
        dbc.Row([
            dbc.Col(
                [
                    dbc.Row(
                        dbc.Button("Добавить камеры", color="secondary", id="add_cameras_button",
                                   style={"width": "100%"}),
                        style={"margin-top": "1rem", "padding": "0rem 1rem"}
                    ),
                    dbc.Row(
                        dcc.Dropdown(
                            id='columns_choose_dropdown',
                            multi=True,
                            options=[
                                {'label': label, 'value': val} for label, val in zip(label_dropdown, value_dropdown)
                            ],
                            value=['id', 'lon', 'lat'],
                        ), style={"margin": "1rem 0rem"}
                    ),
                    dbc.Row(
                        dbc.Col(html.Div(
                            dash_table.DataTable(
                                id='datatable-interactivity',
                                columns=[
                                    {"name": i, "id": i, "selectable": True} for i in
                                    cameras_table[["id", "lat"]].columns
                                ],
                                data=cameras_table[["id", "lat"]].to_dict('records'),
                                sort_action="native",
                                row_selectable="single",
                                selected_columns=[],
                                selected_rows=[0],
                                filter_action="native"
                            ),
                            id='dd-output-container',
                            style={'max-height': 'calc(100vh - 10rem)', "overflow": "scroll", "padding": "0rem 1rem"}
                        ))
                    ),
                ], width=4
            ),
            dbc.Col([
                dbc.Row(
                    # Cards
                    html.Div(id="cards")
                ),

            ], width=8)
        ])

    ],
    id="sku_list",
    className="myTable"
)

hidden_div = html.Div(id="hidden_div", style={"display": "None"})
layout = html.Div([cameras_content, adding_camera_modal, hidden_div])


# обновление картинки и карты при выборе камеры из таблицы
@app.callback(
    Output("cards", "children"),
    [Input('datatable-interactivity', "derived_virtual_data"),
     Input('datatable-interactivity', "derived_virtual_selected_rows")],
    State("datatable-interactivity", "derived_virtual_data")
)
def update_map_image(rows, derived_virtual_selected_rows, data):
    global cameras_table
    if rows is not None:
        cameras_df = pd.DataFrame(data)
        try:
            sorted_cameras_df = cameras_df.merge(cameras_table, on=list(cameras_df.columns))
        except ValueError:
            return dash.no_update
        try:
            dff = sorted_cameras_df.iloc[derived_virtual_selected_rows[0]:derived_virtual_selected_rows[0] + 1, :]
        except IndexError:
            return dash.no_update

        # Обновление карты
        px.set_mapbox_access_token(mapbox_access_token)
        fig = px.scatter_mapbox(pd.DataFrame(dff), lat="lat", lon="lon", zoom=10)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0)
        )

        # Актуальное изображение с камеры
        video_capture = cv2.VideoCapture(dff['source'].iloc[0])
        success, frame = video_capture.read()
        predicted_img = nnapi.make_prediction(frame, dff['id'].iloc[0])

        # Отрисовка предсказанного изображения
        fig_img = go.Figure()
        img_width = 1600
        img_height = 900
        scale_factor = 0.5

        fig_img.update_xaxes(
            visible=False,
            range=[0, img_width * scale_factor]
        )
        fig_img.update_yaxes(
            visible=False,
            range=[0, img_height * scale_factor],
            scaleanchor="x"
        )
        # Add image
        fig_img.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=Image.fromarray(predicted_img))
        )
        fig_img.update_layout(
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
        return [dbc.Row(dcc.Graph(figure=fig,
                                  style={"width": "100%", "padding-right": "1rem", "margin-bottom": "1rem"})),
                dbc.Row(dcc.Graph(figure=fig_img))]

    return dash.no_update


@app.callback(
    Output("modal-centered", "is_open"),
    [Input("add_cameras_button", "n_clicks"), Input("modal_button", "n_clicks")],
    [State("modal-centered", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


new_cameras_content = None


def parse_contents(contents, filename, date):
    global new_cameras_content
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
        new_cameras_content = df


@app.callback(Output('filenames', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    global cameras_table
    global new_cameras_content
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return [html.Span(name) for name in list_of_names]
    return dash.no_update


@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('columns_choose_dropdown', 'value'),
     Input('modal_button', 'n_clicks')])
def update_output(value, button):
    global new_cameras_content
    global cameras_table
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "modal_button" in changed_id:
        updated_cameras_table = pd.concat([new_cameras_content, cameras_table])
        cameras_table = updated_cameras_table

        # TODO : new thread
        dataloader.add_new_cameras_to_db(new_cameras_content)
    if not value:
        return dash.no_update
    elif isinstance(value, list):
        value = value
    else:
        value = [value]
    part_report = cameras_table[value]
    table = dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "selectable": False} for i in part_report.columns
        ],
        style_table={'font-family': "'Arial'"},
        style_header={
            'border': '1px solid #eee',
            'padding': '12px 35px',
            'background': '#4E7AF9',
            'color': '#fff',
            'text-transform': 'uppercase',
            'font-size': '12px',
            'font-weight': 'bold'
        },
        data=part_report.to_dict('records'),
        sort_action="native",
        row_selectable="single",
        selected_columns=[],
        selected_rows=[0],
        filter_action="native"
    )
    return table
