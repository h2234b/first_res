
##使用参数方程
# #通过更改四个控制点坐标更改贝塞尔曲线
# import numpy as np
# from dash import Dash, dcc, html, Input, Output
# import plotly.graph_objs as go

# # 计算三阶贝塞尔曲线
# def bezier_3(t, P0, P1, P2, P3):
#     return (
#         (1 - t) ** 3 * P0 +
#         3 * (1 - t) ** 2 * t * P1 +
#         3 * (1 - t) * t ** 2 * P2 +
#         t ** 3 * P3
#     )

# # 初始化控制点
# control_points = np.array([[0, 0], [1, 2], [3, 2], [4, 0]])

# # 生成贝塞尔曲线
# def generate_bezier(control_points):
#     t = np.linspace(0, 1, 100)
#     curve = np.array([bezier_3(ti, *control_points) for ti in t])
#     return curve

# # 初始化 Dash 应用
# app = Dash(__name__)

# # 初始贝塞尔曲线和控制点数据
# bezier_curve = generate_bezier(control_points)

# # 布局设置
# app.layout = html.Div([
#     html.H1("Interactive Bezier Curve", style={'textAlign': 'center'}),
#     dcc.Graph(
#         id='bezier-graph',
#         config={'scrollZoom': False},
#         style={'height': '600px'}
#     ),
#     html.Div([
#         html.Div([
#             html.Label("Control Point 0 (x, y):"),
#             dcc.Input(id='input-p0x', type='number', value=control_points[0, 0], step=0.1),
#             dcc.Input(id='input-p0y', type='number', value=control_points[0, 1], step=0.1),
#         ], style={'margin': '10px'}),
#         html.Div([
#             html.Label("Control Point 1 (x, y):"),
#             dcc.Input(id='input-p1x', type='number', value=control_points[1, 0], step=0.1),
#             dcc.Input(id='input-p1y', type='number', value=control_points[1, 1], step=0.1),
#         ], style={'margin': '10px'}),
#         html.Div([
#             html.Label("Control Point 2 (x, y):"),
#             dcc.Input(id='input-p2x', type='number', value=control_points[2, 0], step=0.1),
#             dcc.Input(id='input-p2y', type='number', value=control_points[2, 1], step=0.1),
#         ], style={'margin': '10px'}),
#         html.Div([
#             html.Label("Control Point 3 (x, y):"),
#             dcc.Input(id='input-p3x', type='number', value=control_points[3, 0], step=0.1),
#             dcc.Input(id='input-p3y', type='number', value=control_points[3, 1], step=0.1),
#         ], style={'margin': '10px'}),
#     ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '20px'}),
# ])

# # 更新曲线图的回调
# @app.callback(
#     Output('bezier-graph', 'figure'),
#     [Input('input-p0x', 'value'),
#      Input('input-p0y', 'value'),
#      Input('input-p1x', 'value'),
#      Input('input-p1y', 'value'),
#      Input('input-p2x', 'value'),
#      Input('input-p2y', 'value'),
#      Input('input-p3x', 'value'),
#      Input('input-p3y', 'value')]
# )
# def update_curve(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y):
#     global control_points
#     # 更新控制点
#     control_points[0] = [p0x, p0y]
#     control_points[1] = [p1x, p1y]
#     control_points[2] = [p2x, p2y]
#     control_points[3] = [p3x, p3y]
    
#     # 重新生成贝塞尔曲线
#     bezier_curve = generate_bezier(control_points)
    
#     # 创建图表
#     fig = go.Figure()
    
#     # 贝塞尔曲线
#     fig.add_trace(go.Scatter(
#         x=bezier_curve[:, 0],
#         y=bezier_curve[:, 1],
#         mode='lines',
#         name='Bezier Curve',
#         line=dict(color='blue')
#     ))
    
#     # 控制多边形
#     fig.add_trace(go.Scatter(
#         x=control_points[:, 0],
#         y=control_points[:, 1],
#         mode='lines+markers',
#         name='Control Polygon',
#         line=dict(color='red', dash='dash'),
#         marker=dict(size=10, color='green')
#     ))
    
#     # 图表布局
#     fig.update_layout(
#         title="Interactive Bezier Curve",
#         xaxis=dict(title='X', range=[-1, 5]),
#         yaxis=dict(title='Y', range=[-1, 3]),
#         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
#     )
    
#     return fig

# # 启动应用
# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', port=8050, debug=True)



##使用De Casteljau算法
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# 初始控制点
initial_control_points = [[0, 0], [1, 2], [3, 2], [4, 0]]

# De Casteljau 算法
def de_casteljau(control_points, t):
    #通过 De Casteljau 算法计算贝塞尔曲线上的点
    points = np.array(control_points)
    while len(points) > 1:
        points = [(1 - t) * points[i] + t * points[i + 1] for i in range(len(points) - 1)]
    return points[0]

# 生成贝塞尔曲线
def generate_bezier_curve(control_points, num_points=100):
    ts = np.linspace(0, 1, num_points)
    curve = [de_casteljau(control_points, t) for t in ts]
    return np.array(curve)

# Dash 应用
app = dash.Dash(__name__)
app.title = "Interactive Bezier Curve"

# Layout
app.layout = html.Div([
    html.H1("Bezier Curve (Interactive with De Casteljau)", style={'textAlign': 'center'}),
    dcc.Graph(id='bezier-plot', style={'height': '75vh'}),
    html.Div([
        html.Label("Edit Control Points:"),
        dcc.Input(id='input-0', type='text', value="0,0", placeholder="x,y"),
        dcc.Input(id='input-1', type='text', value="1,2", placeholder="x,y"),
        dcc.Input(id='input-2', type='text', value="3,2", placeholder="x,y"),
        dcc.Input(id='input-3', type='text', value="4,0", placeholder="x,y"),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '10px'}),
    html.Div([
        html.Button("Update Curve", id="update-button", n_clicks=0)
    ], style={'textAlign': 'center', 'padding': '10px'}),
])

# 回调：更新曲线
@app.callback(
    Output('bezier-plot', 'figure'),
    [Input('update-button', 'n_clicks')],
    [Input(f'input-{i}', 'value') for i in range(4)]
)
def update_curve(n_clicks, *control_point_values):
    try:
        # 解析用户输入的控制点
        control_points = [list(map(float, cp.split(','))) for cp in control_point_values]
    except ValueError:
        return {
            'data': [],
            'layout': {'title': 'Error: Invalid Control Point Format'}
        }

    # 生成贝塞尔曲线
    curve = generate_bezier_curve(control_points)

    # 绘制控制点、辅助线和曲线
    figure = {
        'data': [
            # 曲线
            {
                'x': curve[:, 0], 'y': curve[:, 1],
                'mode': 'lines', 'line': {'color': 'blue'},
                'name': 'Bezier Curve'
            },
            # 控制点
            {
                'x': [p[0] for p in control_points], 'y': [p[1] for p in control_points],
                'mode': 'markers', 'marker': {'color': 'green', 'size': 10},
                'name': 'Control Points'
            },
            # 控制点连接线
            {
                'x': [p[0] for p in control_points], 'y': [p[1] for p in control_points],
                'mode': 'lines', 'line': {'color': 'red', 'dash': 'dash'},
                'name': 'Control Polygon'
            }
        ],
        'layout': {
            'title': 'Bezier Curve Visualization',
            'xaxis': {'title': 'X'},
            'yaxis': {'title': 'Y'},
            'height': 600
        }
    }
    return figure

# 运行服务器
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
