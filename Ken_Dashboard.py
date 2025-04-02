# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:14:01 2025

@author: lehoa
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

#define function
def style_negative(v, props= ''):
    """Style negative value in dataframe
"""
    try:
        return props if v < 0 else None
    except:
        pass

def style_possitive(v, props=''):
    """ Style possitive value in dataframe"""
    try:
        return props if v>0 else None
    except:
        pass
    
def audience_simple(country):
    if country == "US":
        return 'USA'
    elif country == 'IN':
        return 'India'
    else:
        return 'Other'


#load streamlit
@st.cache_data
##load data
def load_data():
    df_agg = pd.read_csv('Aggregated_Metrics_By_Video.csv').iloc[1:,:]
    df_agg_sub = pd.read_csv('Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_cmt =pd.read_csv('All_Comments_Final.csv')
    df_time = pd.read_csv('Video_Performance_Over_Time.csv')
    
    #rename columns name
    df_agg.columns = ['Video', 'Video title', 'Video publish time', 'Comments added','Shares', 'Dislikes', 'Likes', 'Subscribers lost', 'Subscribers gained','RPM(USD)', 'CPM(USD)', 'Average % viewed', 'Average view duration','Views','Watch time (hours)', 'Subscribers', 'Your estimated revenue(USD)', 'Impressions', 'Impressions ctr(%)' ]
    #re format date time
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'], format="%b %d, %Y")
    #re_format average view duration to hours minute and second
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(lambda x: datetime.strptime(x, '%H:%M:%S'))
    #add column total second duration
    df_agg['Average_duration_sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600 )
    #calculate and add engagement ratio
    df_agg['Engagement_ratio'] = (df_agg['Comments added'] + df_agg['Shares'] + df_agg['Dislikes'] + df_agg['Likes'])/df_agg.Views
    #calculate and add ratio between views and sub gain
    df_agg['Views / sub gained'] = df_agg['Views']/df_agg['Subscribers gained']
    #sort value based on publish date
    df_agg.sort_values('Video publish time', ascending = False, inplace= True)
    return df_agg, df_agg_sub, df_cmt, df_time 

df_agg, df_agg_sub, df_cmt, df_time  = load_data()
#engineer data
#join df_time vs df_agg
df_time_diff = pd.merge(
    df_time, 
    df_agg[['Video', 'Video publish time']], 
    left_on='External Video ID',  # Assuming 'External Video ID' exists in df_time
    right_on='Video',  # Matching column in df_agg
    how='left'  # Keeps all records from df_time and adds matching data from df_agg
)
df_time_diff['Date'] = pd.to_datetime(df_time_diff['Date'], format='mixed', dayfirst=True, errors='coerce')
df_time_diff['days_published'] =(df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days

#get last 12months of data rather than all data
date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months = 12)
df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12mo]

#get daily view data (first 30), median & percentile
views_days = pd.pivot_table(df_time_diff_yr,index = 'days_published', values ='Views',aggfunc = [np.mean, np.median,lambda x: np.percentile(x, 80), lambda x: np.percentile(x, 20)]).reset_index()
views_days.columns = ['days_published', 'mean_views', 'median_views', '80pct_views', '20pct_views']
views_days = views_days[views_days['days_published'].between(0,30)]
views_cumulative = views_days.loc[:,['days_published', 'median_views','80pct_views', '20pct_views']]
views_cumulative[['median_views', '80pct_views', '20pct_views']] = views_cumulative[['median_views', '80pct_views', '20pct_views']].cumsum()


##create dummy dataset df_agg_diff and take data in 1 year 

df_agg_diff = df_agg.copy()

# Define 12-month period
metric_date_12mo = df_agg_diff['Video publish time'].max() - pd.DateOffset(months=12)


# Select only numeric columns (exclude datetime)
numeric_cols = df_agg_diff.select_dtypes(include=['number'])

# Compute median for last 12 months
median_agg = numeric_cols[df_agg_diff['Video publish time'] >= metric_date_12mo].median()

# Identify numeric columns
numeric_index = df_agg_diff.dtypes.isin(['float64', 'int64'])

# Apply transformation (percentage change from median)
df_agg_diff.loc[:, numeric_index] = (df_agg_diff.loc[:, numeric_index] - median_agg).div(median_agg)

## what metrics gonna relevants?
##Difference from baseline
##Percent change by video

#build dashboard on streamlit
add_sidebar = st.sidebar.selectbox('Aggregate or Individual Video',('Aggregate Metrics', 'Individual Video Analysis'))


##total picture
if add_sidebar == 'Aggregate Metrics':
    df_agg_metrics = df_agg[[ 'Video publish time','Comments added','Shares', 'Likes', 'Subscribers','RPM(USD)', 'Average % viewed', 'Average_duration_sec','Engagement_ratio','Views','Views / sub gained']]
    numeric_cols = df_agg_metrics.select_dtypes(include = ['number', 'datetime'])
    metric_date_6mo = numeric_cols['Video publish time'].max() - pd.DateOffset(months =6)
    metric_date_12mo = numeric_cols['Video publish time'].max() - pd.DateOffset(months =12)
    metric_median6mo = numeric_cols[numeric_cols['Video publish time'] >= metric_date_6mo].median() 
    metric_median12mo = numeric_cols[numeric_cols['Video publish time'] >= metric_date_12mo].median() 
    
    col1,col2,col3,col4,col5 = st.columns(5)
    columns = [col1,col2,col3,col4,col5]
    count = 0
    
    for i in metric_median6mo[1:].index:
        with columns[count]:
            delta = (metric_median6mo[i] - metric_median12mo[i])/metric_median12mo[i]
            st.metric(label = i,value = round(metric_median6mo[i], 1), delta ="{:.2%}".format(delta))
            count +=1
            if count >= 5:
                count = 0
    
    
    df_agg_diff['Publish_date'] = df_agg_diff['Video publish time'].apply(lambda x:x.date())
    df_agg_diff_final = df_agg_diff.loc[:, ['Video title', 'Publish_date','Views','Likes', 'Subscribers','Shares', 'Comments added', 'RPM(USD)','Average % viewed', 'Average_duration_sec','Engagement_ratio','Views / sub gained']]
    
    #turn value in to %
    numeric_cols = df_agg_diff_final.select_dtypes(include=['float64', 'int64'])
    df_agg_diff_final[numeric_cols.columns] = df_agg_diff_final[numeric_cols.columns] * 100
    
    #apply on streamlit
    st.dataframe(df_agg_diff_final.style.hide().applymap(style_negative, props ='color:red;').applymap(style_possitive,props ='color:green;').format({col: "{:.2f}%" for col in numeric_cols}))

#Individual video
if add_sidebar == 'Individual Video Analysis':
    videos = tuple(df_agg['Video title'])
    video_select = st.selectbox('Pick a video',videos)
    agg_filtered = df_agg[df_agg['Video title'] == video_select]
    agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_select]
    agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(audience_simple)
    agg_sub_filtered.sort_values('Is Subscribed', inplace = True)
    
    fig = px.bar(agg_sub_filtered, x = 'Views', y = 'Is Subscribed', color = 'Country', orientation='h')
    st.plotly_chart(fig)
    
    agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]
    first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
    first_30 = first_30.sort_values('days_published')
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x = views_cumulative['days_published'], y = views_cumulative['20pct_views'],
                              mode = 'lines',
                              name = '20th percentile', line = dict(color='purple', dash = 'dash')))
    fig2.add_trace(go.Scatter(x = views_cumulative['days_published'], y = views_cumulative['median_views'],
                              mode = 'lines',
                              name = 'median', line = dict(color='green', dash = 'dash')))
    fig2.add_trace(go.Scatter(x = views_cumulative['days_published'], y = views_cumulative['80pct_views'],
                              mode = 'lines',
                              name = '80th percentile', line = dict(color='red', dash = 'dash')))
    fig2.add_trace(go.Scatter(x = first_30['days_published'], y = first_30['Views'].cumsum(),
                              mode = 'lines',
                              name = 'Current Video', line = dict(color = 'firebrick', width = 8)
                              ))
    fig2.update_layout(title = 'Views comparison first 30 days',
                       xaxis_title = 'Days Since Published',
                       yaxis_title = 'Cumulative Views')
    st.plotly_chart(fig2)
    
    
    


































