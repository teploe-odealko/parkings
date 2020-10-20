# dash packages
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# general packages
import pandas as pd
import os
# my packages
from DataLoader import DataLoader
from app import app
from apps import cameras_layout


dataloader = DataLoader()
cameras_table = dataloader.get_cameras_table()
menu = html.Div([
    html.Div([
        dbc.Navbar(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Button(
                            html.Span(className="navbar-toggler-icon"),
                            style={
                                "background": "#343a3f",
                                "border": "none"
                            },
                            id="sidebar-toggle",
                            ),
                            style={"margin-left": "1rem", "display":"flex"}, width={"size": 1}),
                        dbc.Col(dbc.NavbarBrand("PARKING FINDER", className="ml-2"), width={"size": 7, "offset": 1}),

                    ],
                    style={"width": "100%"},
                    justify="between",
                    no_gutters=True,
                ),
            ],
            color="dark",
            dark=True,
            style={"width": "100%", "height": "4rem", "padding": "0rem"}
        ),
        dbc.Row([

            html.Div(dcc.Interval(id="progress-interval", n_intervals=0, interval=500, disabled=True),
                     id="progress_div"),
            dbc.Collapse(dbc.Col(dbc.Progress(id="progress", value=50, style={"height": "4px"}), width=12),
                         id="progress_collapse",
                         is_open=False,
                         style={"width": "100%"})

        ])],
        id="header"),
    html.Div([

        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.NavLink(
                        dbc.Row([
                            dbc.Col("Камеры", width=6),
                            dbc.Col(html.Img(src=app.get_asset_url('camera_icon.svg'), className="sidebar_img"),
                                    width=2)
                        ],
                            justify="between"),
                        href="/page-1", id="page-1-link", active=True),
                    dbc.NavLink(
                        dbc.Row([
                            dbc.Col("Отчеты", width=6),
                            dbc.Col(html.Img(src=app.get_asset_url('report_icon.svg'), className="sidebar_img"), width=2)
                        ],
                            justify="between"),
                        href="/page-2",
                        id="page-2-link"),
                    dbc.NavLink("Настройки", href="/page-3", id="page-3-link"),
                ],
                vertical=True,
                pills=True,
            ),
            id="collapse",
            style={"margin-top": "1rem"}
        ),
    ],
        id="sidebar"),
    html.Div(id="page-content", children=[])
])

app.layout = html.Div([dcc.Location(id="url"), menu])


@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    # print(pathname)
    if pathname == "/page-1/":
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]


# , State("dataset_choose_dropdown", "value")
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1", "/page-1/"]:
        return cameras_layout.layout
    # elif pathname == "/page-2":
    #     return datasets.layout
    # elif pathname == "/page-3":
    #     return settings.layout
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


@app.callback(
    Output("sidebar", "className"),
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className")],
)
def toggle_classname(n, classname):
    if n and classname == "":
        return "collapsed"
    return ""




if __name__ == "__main__":
    app.run_server(port=int(os.environ.get("PORT", 5001)), debug=True, host="0.0.0.0")
