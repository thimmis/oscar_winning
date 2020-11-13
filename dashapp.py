# @Author: Thomas Turner <thomas>
# @Date:   2020-10-08T18:17:09+02:00
# @Email:  thomas.benjamin.turner@gmail.com
# @Last modified by:   thomas
# @Last modified time: 2020-11-13T20:46:35+01:00


import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import pickle
from sklearn.utils import class_weight
from sklearn import metrics
import xgboost as xgb
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
exclude_roles = ['director','producer','executive producer','narrator','host','writer','original music', 'music','screenwriter']
df = pd.read_csv('processed_data.csv')
df = df[~df.ROLE.isin(exclude_roles)]
actor_view_columns = ['TITLE','ROLE','SCORE_1','SCORE_2','NUM_REVIEWS','polarity_mean','objectivity_mean']
prediction_columns = ['AWARD','PRED_gen','PRED_spec']
actor_options = df['ACTOR'].unique()


general_model = pickle.load(open('random_oscar.dat','rb'))

def balancedWeights(label_set):
    classes = label_set.unique()
    classes.sort()
    class_weights = list(class_weight.compute_class_weight('balanced',
                                                           np.unique(label_set),
                                                           label_set.values))

    cw_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
    cw_array = [cw_dict[i] for i in label_set.values]
    return cw_array

specific_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0.1, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.01, max_delta_step=0, max_depth=3,
              min_child_weight=9, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_class=3, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=100,
              reg_lambda=1, scale_pos_weight=None, subsample=0.5,
              tree_method='exact', validate_parameters=1, verbosity=None, silent=1)

app = dash.Dash(__name__, title='Exploring Predictions for Best Actor Winners', external_stylesheets=external_stylesheets)

app.layout = html.Div(id = 'general_div',
                        children = [html.Div(id = 'first_row',
                                            children = [html.Div(id = 'top_block',
                                                                 children=[html.H1("Predicting Best Actor"),
                                                                            html.P("""Here you can compare how the generalized model
                                                                            developed in the Leo_predictions notebook compares to a
                                                                            specified model trained on all but a particular actor's
                                                                            filmography.""")]
                                                                            ),
                                                        html.Div(id = 'middle_block1',
                                                                 children=[html.Label("Choose an actor"),
                                                                            dcc.Dropdown(id = "data_selector",
                                                                                         options = [{'label': i,'value': i}for i in actor_options],
                                                                                         value=actor_options[128],
                                                                                         style = {'width':'50%','align-items':'left','justify-content':'left'})]
                                                                                         )],
                                            className = 'row'),
                                    html.Div(id = 'second_row',
                                            children = [dash_table.DataTable(id = 'actor_data_table',
                                                                             columns = [{'name': i, 'id':i} for i in actor_view_columns],
                                                                             data = [],
                                                                             fixed_rows={'headers': True},
                                                                             style_cell = {'align-items':'left','height': '10px',
                                                                                        # all three widths are needed
                                                                                        'minWidth': '30px', 'width': 'auto', 'maxWidth': 'auto',
                                                                                        'whiteSpace': 'normal'},
                                                                             style_table = {'width':'100%', 'padding':'10px','height': '500px', 'overflowY': 'auto'}),
                                                        dash_table.DataTable(id = 'predictions_table',
                                                                             columns = [{'name': i, 'id':i} for i in prediction_columns],
                                                                             data = [],
                                                                             style_cell = {'align-items':'left','height': '15px',
                                                                                        # all three widths are needed
                                                                                        'minWidth': '15px', 'width': 'auto', 'maxWidth': 'auto',
                                                                                        'whiteSpace': 'normal'},
                                                                             fixed_rows={'headers': True},
                                                                             style_table = {'width':'50%', 'padding':'10px','height': '500px', 'overflowY': 'auto'})],
                                            style = {'columnCount':2},
                                            className = 'row'),
                                    html.Div(id = 'third_row',
                                            children = [dcc.Graph(id = "selected_graph",
                                                                 figure = {'data': []})],
                                            className = 'row')
                        ])

@app.callback(

    [Output('actor_data_table','data'),
    Output('predictions_table','data'),
    Output('selected_graph','figure')],
    Input('data_selector','value')
)
def update_all(value):
    actor_view = df[df['ACTOR']==value].copy()
    actor_view.reset_index(drop=True, inplace=True)
    actor_view_table = actor_view[['TITLE','ROLE','SCORE_1','SCORE_2','NUM_REVIEWS','polarity_mean','objectivity_mean']]
    else_view = df[df['ACTOR'] !=value]
    actor_view_features = actor_view[['SCORE_1','SCORE_2','NUM_REVIEWS','polarity_mean','objectivity_mean','NUMERIC_KEY']]
    actor_view_targets = actor_view['AWARD']
    else_view_features = else_view[['SCORE_1','SCORE_2','NUM_REVIEWS','polarity_mean','objectivity_mean','NUMERIC_KEY']]
    else_view_targets = else_view['AWARD']

    specific_model.fit(else_view_features, else_view_targets, eval_set=[(else_view_features,else_view_targets),(actor_view_features,actor_view_targets)], sample_weight = balancedWeights(else_view_targets), early_stopping_rounds=20)
    specific_preds = pd.DataFrame({'PRED_spec' : specific_model.predict(actor_view_features)})
    specific_preds.reset_index(drop=True, inplace=True)
    general_preds = pd.DataFrame({"PRED_gen":general_model.predict(actor_view[['SCORE_1','SCORE_2','NUM_REVIEWS','polarity_mean','objectivity_mean','NUMERIC_KEY']])})
    general_preds.reset_index(drop=True, inplace=True)
    comp_frame = pd.concat([actor_view,general_preds,specific_preds],axis=1)

    trace1 = {
    "mode" : "markers",
    "type" : "scatter",
    "x" : else_view.SCORE_1,
    "y" : else_view.SCORE_2,
    "name" : 'Training set excluding {}'.format(value),
    "hovertext" : [else_view.TITLE, else_view.ROLE]
    }
    trace2 = {
    "mode" : "markers",
    "type" : "scatter",
    "x" : actor_view[actor_view['AWARD']==2].SCORE_1,
    "y" : actor_view[actor_view['AWARD']==2].SCORE_2,
    "name" : '{} wins'.format(value)
    }
    trace3 = {
    "mode" : "markers",
    "type" : "scatter",
    "x" : actor_view[actor_view['AWARD']==1].SCORE_1,
    "y" : actor_view[actor_view['AWARD']==1].SCORE_2,
    "name" : '{} nominations'.format(value)
    }
    trace4 = {
    "mode" : "markers",
    "type" : "scatter",
    "x" : actor_view[actor_view['AWARD']==0].SCORE_1,
    "y" : actor_view[actor_view['AWARD']==0].SCORE_2,
    "name" : '{} other performances'.format(value)
    }
    scatter1 = go.Scatter(trace1)
    scatter2 = go.Scatter(trace2)
    scatter3 = go.Scatter(trace3)
    scatter4 = go.Scatter(trace4)
    data = [scatter1,scatter2,scatter3,scatter4]
    layout = go.Layout(title = 'Data Breakdown for {}'.format(value))
    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(mode='markers', )

    actor_table = actor_view[['TITLE','ROLE','SCORE_1','SCORE_2','NUM_REVIEWS','polarity_mean','objectivity_mean']].to_dict('rows')

    prediction_table = comp_frame[['AWARD','PRED_gen','PRED_spec']].to_dict('rows')


    return actor_table, prediction_table, fig

if __name__ == '__main__':
    app.run_server(debug=True)
