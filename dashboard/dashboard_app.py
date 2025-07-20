from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from flask import Flask
from datetime import datetime
import socket
import struct

# Initialize Flask app
server = Flask(__name__)

# Initialize Dash app
app = Dash(__name__, server=server)

# Helper function to convert IP to integer
def ip_to_int(ip):
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except socket.error:
        return None  # Handle invalid IPs gracefully

# Load datasets
def load_data():
    fraud_data = pd.read_csv('Data/cleaned_data/Preprocessed_Fraud_Data.csv')
    credit_data = pd.read_csv('Data/cleaned_data/Preprocessed_Creditcard_Data.csv')
    ip_country = pd.read_csv('Data/cleaned_data/Preprocessed_IpAddress_to_Country.csv')
    return fraud_data, credit_data, ip_country

# Data processing functions
def process_ecommerce_data(fraud_data, ip_country):
    fraud_data_cleaned = fraud_data.copy()
    ip_country_cleaned = ip_country.copy()

    fraud_data_cleaned['signup_time'] = pd.to_datetime(fraud_data_cleaned['signup_time'])
    fraud_data_cleaned['purchase_time'] = pd.to_datetime(fraud_data_cleaned['purchase_time'])
    fraud_data_cleaned['purchase_day'] = fraud_data_cleaned['purchase_time'].dt.day_name()
    fraud_data_cleaned['purchase_hour'] = fraud_data_cleaned['purchase_time'].dt.hour
    fraud_data_cleaned['ip_int'] = fraud_data_cleaned['ip_address'].apply(lambda x: ip_to_int(str(int(x))) if pd.notna(x) else None)
    
    ip_country_cleaned['lower_bound_ip_address'] = ip_country_cleaned['lower_bound_ip_address'].astype('int64')
    ip_country_cleaned['upper_bound_ip_address'] = ip_country_cleaned['upper_bound_ip_address'].astype('int64')
    fraud_data_cleaned['ip_int'] = fraud_data_cleaned['ip_int'].astype('int64')
    
    ip_country_cleaned.sort_values('lower_bound_ip_address', inplace=True)
    fraud_data_with_country = pd.merge_asof(
        fraud_data_cleaned.sort_values('ip_int'),
        ip_country_cleaned[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    fraud_data_with_country = fraud_data_with_country[
        (fraud_data_with_country['ip_int'] >= fraud_data_with_country['lower_bound_ip_address']) &
        (fraud_data_with_country['ip_int'] <= fraud_data_with_country['upper_bound_ip_address'])
    ]
    fraud_data_with_country.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1, inplace=True)
    
    return fraud_data_with_country

def create_summary_stats(fraud_data, credit_data):
    ecom_stats = {
        'total_transactions': len(fraud_data),
        'fraud_cases': fraud_data['class'].sum(),
        'fraud_percentage': (fraud_data['class'].sum() / len(fraud_data) * 100).round(2)
    }
    credit_stats = {
        'total_transactions': len(credit_data),
        'fraud_cases': credit_data['Class'].sum(),
        'fraud_percentage': (credit_data['Class'].sum() / len(credit_data) * 100).round(2)
    }
    return ecom_stats, credit_stats

# Load and process data
fraud_data, credit_data, ip_country = load_data()
fraud_data_processed = process_ecommerce_data(fraud_data, ip_country)
ecom_stats, credit_stats = create_summary_stats(fraud_data_processed, credit_data)


# Create the dashboard layout
app.layout = html.Div([
    # Navigation bar
    html.Div([
        html.H1('Fraud Detection Dashboard', className='nav-title'),
        html.P('fraud analytics and insights', className='nav-subtitle')
    ], className='navbar'),

    # Main content container
    html.Div([
        # Summary Statistics Cards
        html.Div([
            html.Div([
                html.Div([
                    html.I(className='fas fa-shopping-cart stat-icon'),
                    html.Div([
                        html.H3('E-commerce Transactions'),
                        html.Div([
                            html.P([
                                html.Span('Total Transactions: ', className='stat-label'),
                                html.Span(f"{ecom_stats['total_transactions']:,}", className='stat-value')
                            ]),
                            html.P([
                                html.Span('Fraud Cases: ', className='stat-label'),
                                html.Span(f"{ecom_stats['fraud_cases']:,}", className='stat-value fraud-value')
                            ]),
                            html.P([
                                html.Span('Fraud Percentage: ', className='stat-label'),
                                html.Span(f"{ecom_stats['fraud_percentage']}%", className='stat-value fraud-value')
                            ])
                        ], className='stat-details')
                    ])
                ], className='stat-card')
            ], className='col-md-6'),

            html.Div([
                html.Div([
                    html.I(className='fas fa-credit-card stat-icon'),
                    html.Div([
                        html.H3('Credit Card Transactions'),
                        html.Div([
                            html.P([
                                html.Span('Total Transactions: ', className='stat-label'),
                                html.Span(f"{credit_stats['total_transactions']:,}", className='stat-value')
                            ]),
                            html.P([
                                html.Span('Fraud Cases: ', className='stat-label'),
                                html.Span(f"{credit_stats['fraud_cases']:,}", className='stat-value fraud-value')
                            ]),
                            html.P([
                                html.Span('Fraud Percentage: ', className='stat-label'),
                                html.Span(f"{credit_stats['fraud_percentage']}%", className='stat-value fraud-value')
                            ])
                        ], className='stat-details')
                    ])
                ], className='stat-card')
            ], className='col-md-6')
        ], className='row stats-container'),

        # Charts Section
        html.Div([
            html.Div([
                html.Div([
                    html.H3('Fraud Trends Over Time', className='chart-title'),
                    dcc.Graph(
                        figure=px.line(
                            fraud_data_processed.groupby(fraud_data_processed['purchase_time'].dt.date)['class'].sum().reset_index(),
                            x='purchase_time',
                            y='class',
                            template='plotly_white'
                        ).update_traces(line_color='#e74c3c')
                    )
                ], className='chart-card')
            ], className='mb-4'),


html.Div([
                html.H3('Geographical Distribution of Fraud', className='chart-title'),
                dcc.Graph(
                    figure=px.choropleth(
                        fraud_data_processed[fraud_data_processed['class'] == 1].groupby('country').size().reset_index(name='count'),
                        locations='country',
                        locationmode='country names',
                        color='count',
                        color_continuous_scale='Reds',
                        template='plotly_white'
                    )
                )
            ], className='chart-card mb-4'),

            # Device and Browser Analysis
            html.Div([
                html.Div([
                    html.H3('Fraud by Device', className='chart-title'),
                    dcc.Graph(
                        figure=px.bar(
                            fraud_data_processed.groupby(['device_id', 'class']).size().unstack().fillna(0),
                            template='plotly_white',
                            color_discrete_sequence=['#2ecc71', '#e74c3c']
                        )
                    )
                ], className='chart-card col-md-6'),

                html.Div([
                    html.H3('Fraud by Browser', className='chart-title'),
                    dcc.Graph(
                        figure=px.bar(
                            fraud_data_processed.groupby(['browser', 'class']).size().unstack().fillna(0),
                            template='plotly_white',
                            color_discrete_sequence=['#2ecc71', '#e74c3c']
                        )
                    )
                ], className='chart-card col-md-6')
            ], className='row mb-4'),

            # Time Patterns
            html.Div([
                html.Div([
                    html.H3('Fraud by Hour of Day', className='chart-title'),
                    dcc.Graph(
                        figure=px.bar(
                            fraud_data_processed[fraud_data_processed['class'] == 1].groupby('purchase_hour').size(),
                            template='plotly_white',
                            color_discrete_sequence=['#e74c3c']
                        ).update_layout(
                            xaxis_title='Hour of Day',
                            yaxis_title='Number of Fraud Cases'
                        )
                    )
                ], className='chart-card col-md-6'),

                html.Div([
                    html.H3('Fraud by Day of Week', className='chart-title'),
                    dcc.Graph(
                        figure=px.bar(
                            fraud_data_processed[fraud_data_processed['class'] == 1].groupby('purchase_day').size(),
                            template='plotly_white',
                            color_discrete_sequence=['#e74c3c']
                        ).update_layout(
                            xaxis_title='Day of Week',
                            yaxis_title='Number of Fraud Cases'
                        )
                    )
                ], className='chart-card col-md-6')
            ], className='row mb-4')
        ], className='charts-container')
    ], className='main-content')
])




app.index_string= '''

<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Fraud Detection Dashboard</title>
    {%favicon%}
    {%css%}
    <style>
        /* Navbar Styling */
        .navbar {
            background-color: #3d405b;
            padding: 1.2rem 2rem;
            color: #fffcf2;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 4px solid #f4a261;
        }
        .nav-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: #e07a5f;
        }
        .nav-subtitle {
            font-size: 1rem;
            color: #e07a5f;
            font-family: 'Roboto', sans-serif;
            font-weight: 300;
        }

        /* Main Content */
        .main-content {
            background-color: #fefae0;
            padding: 2.5rem 3rem;
        }

        /* Summary Statistics Cards */
        .stats-container {
            display: flex;
            justify-content: space-evenly;
            flex-wrap: wrap;
            gap: 1.5rem;
        }

        /* Individual stat card styling */
        .stat-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 6px solid #81b29a;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
            transition: all 0.3s ease;
            max-width: 280px;
            min-width: 220px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        /* Hover effect for stat cards */
        .stat-card:hover {
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.2);
            transform: translateY(-8px);
            background-color: #faf9f9;
            border-color: #f2cc8f;
        }

        /* Stat card icon */
        .stat-icon {
            font-size: 2.5rem;
            color: #3a86ff;
            margin-bottom: 0.8rem;
            transition: color 0.3s ease;
        }
        .stat-card:hover .stat-icon {
            color: #ff006e;
            transform: rotate(360deg);
        }

        /* Stat details */
        .stat-details {
            margin-top: 0.8rem;
        }

        .stat-label {
            font-weight: 700;
            color: #6d6875;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }

        .stat-value {
            font-size: 1.6rem;
            color: #333333;
            font-weight: 600;
        }

        .fraud-value {
            color: #e63946;
            font-weight: bold;
        }

        /* Pulsing hover background effect */
        .stat-card::before {
            content: "";
            position: absolute;
            top: 50%;
            left: 50%;
            width: 250%;
            height: 250%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.3), transparent);
            transform: translate(-50%, -50%) scale(0);
            transition: transform 0.6s ease;
            opacity: 0;
            pointer-events: none;
        }
        .stat-card:hover::before {
            transform: translate(-50%, -50%) scale(1);
            opacity: 1;
        }

        /* Charts Section */
        .charts-container {
            margin-top: 2.5rem;
            padding: 1.5rem;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        }

        .chart-title {
            font-size: 1.4rem;
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            color: #4a4e69;
            margin-bottom: 1rem;
            text-align: left;
        }

        .chart-card {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            transition: box-shadow 0.3s;
            width: 100%;
        }

        .chart-card:hover {
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
            border-left: 5px solid #f4a261;
        }

        /* Responsive adjustments */
        .row {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: space-evenly;
            margin-bottom: 2rem;
        }

        .mb-4 {
            margin-bottom: 2rem;
        }

        /* Custom Colors */
        .plotly_white .main-svg {
            background-color: #faf9f9;
        }

        .col-md-6 {
            max-width: 48%;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>


'''
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080, debug=False)
