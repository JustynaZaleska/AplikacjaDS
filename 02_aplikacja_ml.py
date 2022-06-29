import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd
import pickle

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df =pd.read_csv('./datasets/data_cleaned.csv', index_col=0)

with open('model.pickle', 'rb') as file:
        model = pickle.load(file)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.H3('Model Uczenia Maszynowego - Regresyjny Model Przewidywania Cen Samochodów Używanych'),
        html.H6('Model Lasów Losowych (biblioteka scikit-learn)')
    ], style={'textAlign': 'center'}),
    html.Hr(),
    html.Div([
        html.Label('Podaj rok produkcji samochodu:'),
        dcc.Slider(
            id='slider-1',
            min=df.Year.min(),
            max=df.Year.max(),
            step=1,
            marks={i: str(i) for i in range(df.Year.min(), df.Year.max() + 1)}
        ),
        html.Hr(),
        html.Label('Podaj rozmiar silnika:'),
        dcc.Slider(
            id='slider-2',
            min=0,
            max=6000,
            step=1,
            marks={i: str(i) for i in range(0, 6001, 500)},
            tooltip={'placement': 'bottom'}
        ),
        html.Hr(),
        html.Label('Podaj moc samochodu:'),
        dcc.Slider(
            id='slider-3',
            min=30,
            max=580,
            step=1,
            marks={i: str(i) for i in range(30, 581, 50)},
            tooltip={'placement': 'bottom'}
        ),
        html.Br(),
        html.Label('Podaj liczbę pasażerów:'),
        html.Div([
            dcc.Dropdown(
                id='dropdown-1',
                options=[{'label': i, 'value': i} for i in [2, 4, 5, 6, 7, 8, 9, 10]]
            )
        ], style={'width': '20%', 'textAlign': 'left'}),
        html.Br(),
        html.Label('Podaj typ paliwa:'),
        html.Div([
            dcc.Dropdown(
                id='dropdown-2',
                options=[{'label': i, 'value': i} for i, j in zip(['Diesel', 'Benzyna', 'CNG', 'LPG', 'Elektryczny'],
                                                                 ['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric'])]
            )
        ], style={'width': '20%', 'textAlign': 'left'}),
        html.Br(),
        html.Label('Podaj typ przekładni:'),
        html.Div([
            dcc.Dropdown(
                id='dropdown-3',
                options=[{'label': i, 'value': i} for i, j in zip(['Manualna', 'Autmatyczna'],
                                                                  ['Manual', 'Automatic'])]
            )
        ], style={'width': '20%', 'textAlign': 'left'}),

        html.Div([
            html.Hr(),
            html.H3('Predykcja na podstawie modelu'),
            html.Hr(),
            html.H4('Podałeś parametry: '),
            html.Div(id='div-1'),
            html.Div(id='div-2'),
            html.Hr()
        ], style={'margin': '0 auto', 'textAlign': 'center'})
    ], style={'width': '80%', 'textAlign': 'left', 'margin': '0 auto', 'fontSize': 22})
])

if __name__ == '__main__':
    app.run_server(debug=True)
