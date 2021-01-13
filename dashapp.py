# @Author: Thomas Turner <thomas>
# @Date:   2020-10-08T18:17:09+02:00
# @Email:  thomas.benjamin.turner@gmail.com
# @Last modified by:   thomas
# @Last modified time: 2020-12-28T15:53:47+01:00


import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from plotly import graph_objs as go
from plotly import express as px
import dash_table
import pandas as pd
import numpy as np


app = dash.Dash(__name__, title='Exploring Predictions for Best Actor Winners', external_stylesheets=[dbc.themes.LUX])

server = app.server


#reading in data used to train and test models
df_data = pd.read_csv('https://raw.githubusercontent.com/thimmis/data_set/master/best_actor_hd.csv', sep=',')

#define dataset of all features used in the training of the new models
data_headers_all = ['score_1','score_2','metascore','avg_vote','weighted_average_vote','mean_vote','median_vote',
        'males_allages_avg_vote','males_allages_votes','females_allages_avg_vote','females_allages_votes',
       'top1000_voters_rating','top1000_voters_votes','us_voters_rating','us_voters_votes','non_us_voters_rating',
       'non_us_voters_votes','age_at_performance','year','height','ordering','num_reviews','polarity_mean','objectivity_mean',
       'budget','worlwide_gross_income','duration','numeric_key']
#break down features into lists from the new and old data sets
data_headers_original = ['score_1','score_2','num_reviews','polarity_mean','objectivity_mean','numeric_key']
data_headers_new = ['metascore','avg_vote','weighted_average_vote','mean_vote','median_vote',
        'males_allages_avg_vote','males_allages_votes','females_allages_avg_vote','females_allages_votes',
       'top1000_voters_rating','top1000_voters_votes','us_voters_rating','us_voters_votes','non_us_voters_rating',
       'non_us_voters_votes','age_at_performance','year','height','ordering',
       'budget','worlwide_gross_income','duration',]

#read in the datasets containing the predictions for each model and the overlap for all actors/titles
df_predictions = pd.read_csv('https://raw.githubusercontent.com/thimmis/data_set/master/performance_predictions.csv', sep=',')

#merge the predictions and feature information for faceting the data based on the models
df_comprehensive = pd.merge(df_predictions,df_data, how='outer',on=['actor','title','year','award','age_at_performance'])
df_comprehensive = df_comprehensive.rename(columns={'award':'observed','gm_preds':'general_model','sm_preds':'special_model','mean_pred':'prediction_overlap'})

#create lists for all years and ages at time of performance to populate sliders for fitering data.
years_list = df_predictions.year.unique().tolist()
age_list = df_predictions.age_at_performance.unique().tolist()

#provide the column headers and data for exploring the datatables of the actor/title information by actor or year
pred_headers = ['year','title','actor','observed','general_model','special_model','model_overlap']
df_confidence = pd.read_csv('https://raw.githubusercontent.com/thimmis/data_set/master/prediction_metrics.csv', sep=',')
metric_headers = df_confidence.columns.tolist()[1:]




#static banner element for the page layout.
card_title = dbc.Card(
    [
    dbc.CardBody(
        [
        html.H3("'... And the Award for Best Actor goes to...'",
        className='card-title'),
        html.H6("Exploring the Actual and predicted winners",
        className='card-subtitle')
        ]
    ),
    ],
    outline=False,
)

"""
The page information framework. This card provides two tabs that dynamically populate the empty html.Div located below the tabs. When the Predictions tab is selected the view allows the user to use sliders to dynamically change the faceted information for a given year compared to a given model choice. If the user selects Data they can choose which set of features to graph as well as which axis the two features should be on as well as an additional option to facet the data for all model types.
"""
card_explorer = dbc.Card(
    [
        dbc.CardBody(
            [
                dcc.Tabs(id='data-pred-tabs',value='predictions',children=[
                    dcc.Tab(label='Predictions', value='predictions'),
                    dcc.Tab(label='Data', value='data')
                ],
                persistence=True),
                html.Div(id='layout-body')
            ]
        )

    ]

)

#The first component of the predictions tab: lets the user select a number of years and age at performance to view, as well as which model type to compare the observed awards against. These will dynamically update the four bar charts containing the yearly information as well as the total frequency..
card_selector1 = dbc.Card(
    [
    dbc.CardBody(
        [
        html.Label("Filter by the Year", className="control_label"),
        dcc.RangeSlider(
            id="year_slider",
            min=1927,
            max=max(years_list)-1,
            value = [1927,max(years_list)-1],
            tooltip = { 'always_visible': False },
            persistence=True,
        ),
        html.Label("Filter by Age"),
        dcc.RangeSlider(
            id="age_slider",
            min=min(age_list),
            max=max(age_list),
            value=[min(age_list),max(age_list)],
            tooltip = { 'always_visible': False },
            persistence=True,

        ),
        html.Label("Select a prediction type for comparison"),
        dcc.Dropdown(id = "data_selector",
                     options = [{'label': 'General Model','value': 'gm_preds'},
                     {'label': 'Special Model','value': 'sm_preds'},
                     {'label': 'Overlap','value': 'mean_pred'}],
                     value='gm_preds',
                     clearable=False,
                     persistence=True),
        ]
    ),
    ],

)

"""
The second body component of the Predictions tab. card_multigraphs1 contains an empty graph that is updated by the two sliders and dropdown menu from the first component and displays the frequency of awards by year and provides a facet by the selected model type to the observed awards.
"""

card_multigraphs1 = dbc.Card(
    [
    dbc.CardBody(
        [
                html.Label('Breakdown by Year'),
                dcc.Graph(id = "time_series1",
                    figure = {
                        'data': [],
                        })
        ]
    ),
    ],
)
"""
The third body component of the Predictions tab. card_multigraphs2 contains an empty graph that is updated by the two sliders and dropdown menu from the first component.
"""
card_multigraphs2 = dbc.Card(
    [
    dbc.CardBody(
        [
                html.Label('Frequency of Awards'),
                dcc.Graph(id="frequency_plot1",
                      figure = {'data': []})
        ],

    ),
    ],
)
"""
Fourth body component. Prodivded an additional selection tab for the user to choose between viewing the title and actor information for a given year, or view all information for an actor. Based on their selection this dynamically populates the tab with further choices. Default is by year.
"""
card_selector2 = dbc.Card(
    [
        dbc.CardBody(
            [
                html.Label('View predictions by year or by actor'),
                dcc.Tabs(id='tabs',value='year',children=[
                    dcc.Tab(label='Year', value='year'),
                    dcc.Tab(label='Actor', value='actor')
                ],
                persistence=True
                ),
                html.Div(id='tabs-content')
            ]
        ),
    ],
    color='light',
    inverse=False,
    outline=False,
)


"""
Fifth body component for the Predictions Tab. This contains the datatable that will display the actor or title information depending on how the user chooses.
"""

card_tabledata = dbc.Card(
    [
    dbc.CardBody(
        [
            html.Div(id='table_main')
        ]
    ),
    ],
    color='light',
    inverse=False,
    outline=False,
)

"""
Sixth Body component for the Predictions Tab. This displays the accuracy and roc scores for each actor's predictions for each model.
"""
card_tableconfidence = dbc.Card(
    [
    dbc.CardBody(
        [
            html.Div(id='table_secondary')

        ]
    ),
    ],
    color='light',
    inverse=False,
    outline=False,
)

"""
The first body component for the Data Tab. This particular component sets up the graph for exploring the features used to train the models and for making the predictions.
"""
card_data = dbc.Card(
    [
        dbc.CardBody(
            [
                dcc.Graph(id='data_figure',
                figure = {'data':[]})
            ],
            className='row'
        ),
    ],
    color='light',
    inverse=False,
    outline=False,
)

"""
The second body component for the Data Tab. This component dynamically updates the two drop down menus based on which radio item is selected and which feature is selected for the x-axis. Additionally it allows the user to facet the graph for all model types by user selection, the graph is not updated until the button is pushed.
"""
card_data_selector = dbc.Card(
    [
        dbc.CardBody(
            [
                html.Label('Select which dataset to explore:'),
                dcc.RadioItems(id='radio_subsets',
                    options = [
                        {'label':'Entire set', 'value':'entire'},
                        {'label':'Original set','value':'original'},
                        {'label':'New set','value':'new'}
                    ],
                    value='entire',
                    persistence=True,
                    labelStyle={'display': 'block','margin-left':'10px'}
                ),
                html.Div(id='dual_droppers'),
                html.P('Facet scatter plot by all award types?'),
                dcc.RadioItems(id='facet_data',
                    options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'}
                ],
                    value='yes',
                    persistence=True,
                    labelStyle={'display': 'inline-block','margin-left':'10px'}
                ),
                html.P(' '),
                html.Button('Graph',id='graph_button')
            ]
        ),
    ],
    color='light',
    inverse=False,
    outline=False,
)

"""
This is the entire bootstrap layout of the web application.
"""
app.layout = html.Div(
    [
    dbc.Row([dbc.Col(card_title, width=6)],justify='center'),
    dbc.Row([dbc.Col(card_explorer,width=11)],justify='center')
    ],
    className='dash-bootstrap'
)
#helper functions

def filter_dataframe(frame, year_slider, age_slider):
    '''
    Filters a select Pandas DataFrame by an upper and lower bounds for both Year and Age
    '''
    dff = frame[
        (frame['year']>=year_slider[0])
        &(frame['year']<=year_slider[1])
        &(frame['age_at_performance']>=age_slider[0])
        &(frame['age_at_performance']<=age_slider[1])
    ]
    return dff

def generate_table(frame, select_year,radio_option):
    '''
    Generates information to populate a dash_table.DataTable when the user wants to look at a specific year, pulls the correct possible frame from a dictionary based on whether they want to view all non-zero predictions for a given year.
    '''
    table_dict = {'no': frame[(frame.year == select_year)
                &(frame.award != 0)],
                'yes': frame[(frame.year == select_year)
                            &(frame.gm_preds != 0)
                            &(frame.sm_preds != 0)]}
    return table_dict[radio_option]


def get_actor_data(frame, select_actor):
    '''
    Generates a copy of a pd.DataFrame filtered by a select actor.
    '''
    return(frame[frame.actor == select_actor])

def get_confidence(frame, names):
    '''
    returns a copy of the df_confidence frame using a list of actor names that are present in the dash_table.
    '''
    cdff = frame[frame.actor.isin(names)]
    return cdff

def get_actor_specifics(frame, select_actor):
    '''
    Computes the medal frequencies for actor(s) and melts data into long format
    '''
    temp_dff = frame[frame.actor==select_actor].agg({i:'value_counts' for i in frame[['award','gm_preds','sm_preds','mean_pred']]}).reset_index().rename(columns={'index':'type','award':'observed','gm_preds':'general','sm_preds':'special','mean_pred':'overlap'})
    mdff = pd.melt(temp_dff, id_vars=['type'],value_vars=['observed','general','special','overlap'])
    return mdff

def remove_item_from_list(the_item,the_list):
    '''
    Takes an item and a list and returns a new list without that particular item in it.
    '''
    new_list = the_list.copy()
    try:
        new_list.remove(the_item)
    except Exception as e:
        pass
    return new_list

#Tab selector to explore the predictions or the underlying data
@app.callback(
    Output('layout-body','children'),
    Input('data-pred-tabs','value')
)
def swap_tabs(which_tab):
    swap_dict = {'predictions':(dbc.Row([dbc.Col(card_selector1, width=2),
            dbc.Col(card_multigraphs1,width=6),
            dbc.Col(card_multigraphs2,width=4)]),dbc.Row([dbc.Col(card_selector2,width=2),
                dbc.Col(card_tabledata, width=6),
                dbc.Col(card_tableconfidence,width=4)])),
                'data':(dbc.Row([dbc.Col(card_data_selector,width=2),
                        dbc.Col(card_data,width=10)]))}
    return swap_dict[which_tab]


#plot frequencies by year for each category
@app.callback(
    Output('time_series1','figure'),
    [Input('year_slider','value'),
    Input('age_slider','value'),
    Input('data_selector','value')],
    )
def update_multi_plot(year_slider, age_slider, data_selector):
    dff = filter_dataframe(df_predictions, year_slider, age_slider)
    dfg = dff.groupby(['year']).agg({i:'value_counts' for i in dff[['award','gm_preds','sm_preds','mean_pred']]}).reset_index().rename(columns={'level_0':'year','level_1':'type','award' :'observed'}).fillna(0)
    dfg2 = pd.melt(dfg, id_vars=['year','type'],value_vars=['observed',data_selector])

    fig1 = px.bar(dfg2, x="year", y='value' ,color="type", facet_col='variable', barmode='stack', facet_col_wrap=1)
    fig1.update_layout(margin=dict(l=20, r=20, t=20, b=20),height=600)

    return fig1

#plot total frequency of value counts
@app.callback(
Output('frequency_plot1','figure'),
[Input('year_slider','value'),
Input('age_slider','value'),
Input('data_selector','value')]
)
def update_frequencies(year_slider, age_slider, data_selector):
    dff = filter_dataframe(df_predictions, year_slider, age_slider).rename(columns={'award':'observed'})
    dfg = dff.agg({i:'value_counts' for i in dff[['observed','gm_preds','sm_preds','mean_pred']]}).reset_index().rename(columns={'index':'type','award':'observed'})
    mdfg = pd.melt(dfg, id_vars=['type'],value_vars=['observed', data_selector])
    fig2 = px.bar(mdfg, x='type', y='value', color='type', facet_col='variable',facet_col_wrap=1)
    fig2.update_layout(margin=dict(l=20, r=20, t=20, b=20),height=600)

    return fig2

#update second row Dropdown content
@app.callback(
[Output('tabs-content','children'),
Output('table_main','children'),
Output('table_secondary','children')],
Input('tabs','value')
)
def render_content(which_tab):

    option_dict = {'year' : (html.Div([
        html.P(),
        dcc.Dropdown(id="predictions_dropdown",
        options=[{'label':i, 'value':i} for i in df_predictions.year.sort_values(ascending=False).unique().tolist()[1:]],
        value=2019,
        clearable=False,
        persistence=True),
        html.P(' '),
        html.Label('Show additional Predictions?'),
        dcc.RadioItems(id='radio_selector',
            options=[
            {'label': 'Yes', 'value': 'yes'},
            {'label': 'No', 'value': 'no'}
        ],
        value='no',
        persistence=True,
        labelStyle={'display': 'inline-block','margin-left':'10px'})
    ]),html.Div([
            html.Label('Observations and Predicitons by Year'),
            dash_table.DataTable(id = "year_table",
                                columns = [{"name" : i, "id":i} for i in pred_headers],
                                data = [],
                                fixed_rows={'headers': True},
                                style_cell = {'textAlign':'left','height': '10px',
                                           'minWidth': '15px', 'width': '15px', 'maxWidth': '40px',
                                           'whiteSpace': 'normal'},
                                style_table = {'padding':'10px','height': '400px', 'overflowY': 'auto'},
                                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                                persistence=True)
    ]), html.Div([
            html.Label('Model Accuracy and ROC AUC Scores'),
            dash_table.DataTable(id = "year_confidence",
                                columns = [{"name": i, "id": i} for i in metric_headers],
                                data = [],
                                fixed_rows={'headers': True},
                                style_cell = {'textAlign':'left','height': '10px',
                                           'minWidth': '15px', 'width': '15px', 'maxWidth': '40px',
                                           'whiteSpace': 'normal'},
                                style_table = {'padding':'10px','height': '400px', 'overflowY': 'auto'},
                                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                                persistence=True)
    ])), 'actor': (html.Div([
        html.P(),
        dcc.Dropdown(id='actor_dropdown',
        options=[{'label':i ,'value':i} for i in df_predictions.actor.sort_values(ascending=True).unique().tolist()],
        value='Leonardo DiCaprio',
        clearable=False,
        persistence=True),
    ]),html.Div([
            html.Label('Breakdown of Awards and Predictions by Actor'),
            dash_table.DataTable(id = "actor_table",
                                columns = [{"name" : i, "id":i} for i in pred_headers],
                                data = [],
                                fixed_rows={'headers': True},
                                style_cell = {'textAlign':'left','height': '10px',
                                           'minWidth': '15px', 'width': '15px', 'maxWidth': '40px',
                                           'whiteSpace': 'normal'},
                                style_table = {'padding':'10px','height': '400px', 'overflowY': 'auto'},
                                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                                persistence=True)
    ]),html.Div([
            html.Label('Model Accuracy and ROC AUC scores'),
            dash_table.DataTable(id = "actor_confidence",
                                columns = [{"name": i, "id": i} for i in metric_headers],
                                data = [],
                                fixed_rows={'headers': True},
                                style_cell = {'textAlign':'left','height': '10px',
                                           'minWidth': '15px', 'width': '15px', 'maxWidth': '40px',
                                           'whiteSpace': 'normal'},
                                style_table = {'padding':'10px','height': '110px', 'overflowY': 'auto'},
                                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                                persistence=True),
            html.Label('Actor Award Frequency'),
            dcc.Graph(id='frequency_plot3',
              figure = {'data': []})
    ]))}
    return option_dict[which_tab]


#filter data frame and return
@app.callback(
[Output('year_table','data'),
Output('year_confidence','data')],
[Input('predictions_dropdown','value'),
Input('radio_selector','value')]
)
def update_yearview(select_year,radio_option):
    tdff_no = generate_table(df_predictions, select_year,radio_option).rename(columns={'award':'observed','gm_preds':'general_model','sm_preds':'special_model','mean_pred':'model_overlap'})
    names_no = tdff_no.actor.unique().tolist()
    cdff_no = get_confidence(df_confidence, names_no)[metric_headers]
    return tdff_no.to_dict('rows'), cdff_no.to_dict('rows')


#update data in table when actor is selected
@app.callback(
[Output('actor_table','data'),
Output('actor_confidence','data'),
Output('frequency_plot3','figure')],
Input('actor_dropdown','value')
)
def update_actorview(select_actor):
    actor_dff = get_actor_data(df_predictions,select_actor).rename(columns={'award':'observed','gm_preds':'general_model','sm_preds':'special_model','mean_pred':'model_overlap'})
    actor_name = actor_dff.actor.tolist()
    cdff = get_confidence(df_confidence, actor_name)[metric_headers]

    mdff = get_actor_specifics(df_predictions, select_actor)
    fig3 = px.bar(mdff, x='type', y='value',color='variable',barmode='group')
    fig3.update_layout(margin=dict(l=20, r=20, t=20, b=20),height=290)
    return actor_dff.to_dict('rows'), cdff.to_dict('rows'), fig3

#return the proper drop downs for both x and y scatter information for the subset
@app.callback(
Output('dual_droppers','children'),
Input('radio_subsets','value')
)
def populate_data_dropdowns(which_subset):
    set_dict = {
                'entire' : data_headers_all,
                'original' : data_headers_original,
                'new' : data_headers_new
                }
    return html.Div([html.Label('Choose x-variable:'),
                dcc.Dropdown(id='x_selector',
                    options=[{'label':i, 'value':i} for i in set_dict[which_subset]],
                    value=set_dict[which_subset][0],
                    clearable=False,
                    persistence=True),
                html.P(' '),
                html.Label('Choose y-variable:'),
                html.Div(id='y_selector_container'),
                ])


#exclude the selected x variable from the y variable list
@app.callback(
Output('y_selector_container','children'),
[Input('x_selector','value'),
Input('radio_subsets','value')]
)
def create_y_dropdown(invalid_choice, which_subset):
    set_dict = {
                'entire' : remove_item_from_list(invalid_choice, data_headers_all),
                'original' : remove_item_from_list(invalid_choice, data_headers_original),
                'new' : remove_item_from_list(invalid_choice,data_headers_new)
                }

    return dcc.Dropdown(
                id='y_selector',
                options=[{'label':i,'value':i} for i in set_dict[which_subset]],
                value=set_dict[which_subset][0],
                clearable=False,
                persistence=True
                )

#populate the graph with a scatterplot of the two selected variables
@app.callback(
Output('data_figure','figure'),
Input('graph_button','n_clicks'),
[State('x_selector','value'),
State('y_selector','value'),
State('facet_data','value')]
)
def populate_data_graph(n, x_col, y_col, faceter):
    if faceter == 'no':
        fig4 = px.scatter(df_data, x=x_col, y=y_col, color='award', hover_name='title',
        hover_data=['year','actor','age_at_performance'])
        fig4.update_layout(margin=dict(l=20, r=20, t=20, b=20),height=900, width=1265)
        return fig4
    elif faceter == 'yes':
        fig5 = px.scatter(
            pd.melt(df_comprehensive, id_vars=['actor','title',x_col,y_col],value_vars=['observed','general_model','special_model','prediction_overlap']),
            x=x_col, y=y_col, facet_col='variable', color='value', facet_col_wrap=2, hover_name='title',hover_data=['actor'])
        fig5.update_layout(margin=dict(l=20, r=20, t=20, b=20),height=900, width=1265)
        return fig5


if __name__ == '__main__':
    app.run_server(debug=False)
