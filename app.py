import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os

st.set_page_config(layout='wide', page_title='Lead Conversion: Urgent vs Normal')

@st.cache_data
def load_data():
    # Prefer cleaned CSV then Excel
    if os.path.exists('data_fy25_cleaned.csv'):
        df = pd.read_csv('data_fy25_cleaned.csv', parse_dates=['created_on'], low_memory=False)
    elif os.path.exists('data_fy25.xlsx'):
        df = pd.read_excel('data_fy25.xlsx', engine='openpyxl')
    else:
        raise FileNotFoundError('No data file found. Put data_fy25_cleaned.csv or data_fy25.xlsx in working folder')
    return df

def normalize_priority(x):
    if pd.isna(x):
        return 'Normal'
    s = str(x).strip().lower()
    if s in ('urgent','high','p1','1','critical','true') or 'urgent' in s:
        return 'Urgent'
    return 'Normal'

def prepare(df):
    df = df.copy()
    if 'created_on' in df.columns:
        df['created_on'] = pd.to_datetime(df['created_on'], errors='coerce')
    if 'yyyymm' not in df.columns and 'created_on' in df.columns:
        df['yyyymm'] = df['created_on'].dt.strftime('%Y-%m')
    else:
        if 'yyyymm' in df.columns:
            df['yyyymm'] = df['yyyymm'].astype(str)
    # normalized fields
    if 'priority' in df.columns:
        df['_priority_norm'] = df['priority'].apply(normalize_priority)
    else:
        df['_priority_norm'] = 'Normal'
    if 'status' in df.columns:
        df['_is_qualified'] = df['status'].astype(str).str.strip().str.lower() == 'qualified'
    else:
        df['_is_qualified'] = False
    return df

def detect_hp_col(df):
    candidates = [c for c in df.columns if 'hp_lead' in c.lower() or 'lead_count' in c.lower() or c.lower()=='lead_count']
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def apply_filters(df, markets=None, countries=None, months=None, priorities=None, business_groups=None, statuses=None):
    mask = pd.Series(True, index=df.index)
    if markets:
        mask &= df['market'].isin(markets)
    if countries:
        mask &= df['country'].isin(countries)
    if months and 'yyyymm' in df.columns:
        mask &= df['yyyymm'].isin(months)
    if priorities:
        mask &= df['_priority_norm'].isin(priorities)
    if business_groups:
        if 'business_group' in df.columns:
            mask &= df['business_group'].isin(business_groups)
    if statuses:
        if 'status' in df.columns:
            mask &= df['status'].isin(statuses)
    return df.loc[mask].copy()

def df_to_bytes(df):
    b = BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

# Load and prepare data
try:
    df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

df = prepare(df)
hp_col = detect_hp_col(df)
# Remove Greater China from the dataset entirely so it doesn't appear in filters or charts
if 'market' in df.columns:
    df = df.loc[df['market'].astype(str).str.strip().str.lower() != 'greater china'].copy()

# Sidebar filters
st.sidebar.header('Filters')
market_opts = []
country_opts = []
month_opts = []
priority_opts = sorted(df['_priority_norm'].unique().tolist())
business_opts = []
if 'market' in df.columns:
    market_opts = sorted(df['market'].dropna().unique().tolist())
if 'country' in df.columns:
    country_opts = sorted(df['country'].dropna().unique().tolist())
if 'yyyymm' in df.columns:
    month_opts = sorted(df['yyyymm'].dropna().unique().tolist())
if 'business_group' in df.columns:
    business_opts = sorted(df['business_group'].dropna().unique().tolist())

sel_markets = st.sidebar.multiselect('Market', options=market_opts, default=None)
sel_countries = st.sidebar.multiselect('Country', options=country_opts, default=None)
sel_months = st.sidebar.multiselect('Month (YYYY-MM)', options=month_opts, default=None)
sel_priorities = st.sidebar.multiselect('Priority', options=priority_opts, default=None)
sel_business = st.sidebar.multiselect('Business Group', options=business_opts, default=None)

if 'status' in df.columns:
    status_opts = sorted(df['status'].dropna().unique().tolist())
else:
    status_opts = []

sel_status = st.sidebar.multiselect('Status', options=status_opts, default=None)

st.sidebar.markdown('---')
st.sidebar.write('Data rows: %s' % f"{len(df):,}")

# Apply filters (treat empty selection as all)
f_markets = sel_markets if sel_markets else None
f_countries = sel_countries if sel_countries else None
f_months = sel_months if sel_months else None
f_priorities = sel_priorities if sel_priorities else None
f_business = sel_business if sel_business else None
f_status = sel_status if sel_status else None

filtered = apply_filters(df, markets=f_markets, countries=f_countries, months=f_months, priorities=f_priorities, business_groups=f_business, statuses=f_status)

st.title('Lead Conversion: Urgent vs Normal')
st.markdown('Interactive Streamlit dashboard: compare conversion between Urgent and Normal priorities. Use the sidebar filters to slice the data.')

# --- Main KPIs and priority comparison ---
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    if hp_col:
        total = int(round(filtered[hp_col].sum()))
    else:
        total = int(len(filtered))
    st.metric('Total leads', f'{total:,}')
with col2:
    if hp_col:
        qualified = int(round(filtered.loc[filtered['_is_qualified'], hp_col].sum()))
    else:
        qualified = int(filtered['_is_qualified'].sum())
    st.metric('Qualified', f'{qualified:,}')
with col3:
    conv_pct = (qualified / total * 100) if total else 0
    st.metric('Conversion %', f'{conv_pct:.1f}%')
with col4:
    if 'tov' in filtered.columns and pd.api.types.is_numeric_dtype(filtered['tov']):
        avg_tov = filtered['tov'].mean()
        # show monetary-like integer without decimals
        st.metric('Average TOV', f"{int(round(avg_tov)):,}")
    else:
        st.metric('Average TOV', 'N/A')

st.markdown('### Conversion % by Priority')
if not filtered.empty:
    if hp_col:
        agg = filtered.groupby('_priority_norm')[hp_col].agg(total_leads='sum')
        agg['qualified_leads'] = filtered.loc[filtered['_is_qualified']].groupby('_priority_norm')[hp_col].sum()
    else:
        agg = filtered.groupby('_priority_norm').size().rename('total_leads').to_frame()
        agg['qualified_leads'] = filtered.loc[filtered['_is_qualified']].groupby('_priority_norm').size()
    agg = agg.fillna(0)
    agg['conversion_pct'] = 100 * agg['qualified_leads'] / agg['total_leads'].replace({0: np.nan})
    for p in ['Urgent', 'Normal']:
        if p not in agg.index:
            agg.loc[p] = {'total_leads':0,'qualified_leads':0,'conversion_pct':0}
    agg = agg.loc[['Urgent','Normal']]
    fig = px.bar(agg.reset_index(), x='_priority_norm', y='conversion_pct',
                 text=agg['conversion_pct'].fillna(0).map(lambda v: f"{v:.1f}%"),
                 labels={'_priority_norm':'Priority','conversion_pct':'Conversion %'}, height=460)
    # configure text inside bars for percentages
    fig.update_traces(textposition='inside')
    # compute y-axis max
    y_max = max(10, (agg['conversion_pct'].max() or 0) * 1.3)
    fig.update_layout(yaxis=dict(range=[0, y_max]))
    # add two-line annotations (Qualified / Total) inside each bar for clearer absolute numbers
    df_agg = agg.reset_index()
    for _, row in df_agg.iterrows():
        pr = row['_priority_norm']
        conv = row['conversion_pct'] if pd.notna(row['conversion_pct']) else 0
        qualified_val = int(row['qualified_leads']) if pd.notna(row['qualified_leads']) else 0
        total_val = int(row['total_leads']) if pd.notna(row['total_leads']) else 0
        # place annotation roughly at half the bar height
        y_pos = conv * 0.5
        ann_text = f"Q: {qualified_val:,}<br>Total: {total_val:,}"
        fig.add_annotation(x=pr, y=y_pos,
                           text=ann_text, showarrow=False,
                           font=dict(size=11, color='white'), align='center',
                           bgcolor='rgba(0,0,0,0.35)', bordercolor='rgba(0,0,0,0.1)')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No data available for selected filters.')

st.markdown('### Qualified / Open / Disqualified by Priority (stacked)')
if not filtered.empty:
    # If we have a status column, split non-qualified rows into Open and Disqualified where possible
    if 'status' in filtered.columns:
        def map_segment(row):
            if row.get('_is_qualified'):
                return 'Qualified'
            s = str(row.get('status', '')).strip().lower()
            if 'disqual' in s:
                return 'Disqualified'
            if 'open' in s:
                return 'Open'
            return 'Other'

        seg_df = filtered.copy()
        seg_df['segment'] = seg_df.apply(map_segment, axis=1)

        if hp_col:
            pivot = seg_df.groupby(['_priority_norm', 'segment'])[hp_col].sum().unstack(fill_value=0)
        else:
            pivot = seg_df.groupby(['_priority_norm', 'segment']).size().unstack(fill_value=0)

        # Ensure consistent order of priorities and segments
        for p in ['Urgent', 'Normal']:
            if p not in pivot.index:
                pivot.loc[p] = 0
        for seg in ['Qualified', 'Open', 'Disqualified', 'Other']:
            if seg not in pivot.columns:
                pivot[seg] = 0
        pivot = pivot[['Qualified', 'Open', 'Disqualified', 'Other']].loc[['Urgent', 'Normal']]

        fig2 = go.Figure()
        colors = {'Qualified':'green', 'Open':'orange', 'Disqualified':'red', 'Other':'lightgray'}
        # show absolute values with percentages in brackets inside stacked segments
        # Calculate totals per priority for percentage calculation
        row_totals = pivot.sum(axis=1)
        for seg in ['Qualified', 'Open', 'Disqualified', 'Other']:
            seg_vals = pivot[seg].astype(float).fillna(0)
            abs_vals = seg_vals.map(lambda v: int(round(v)))
            # Calculate percentages for each segment relative to its priority row total
            pct_vals = (seg_vals / row_totals * 100).fillna(0).map(lambda v: f"{v:.1f}%")
            # Format as: 1,234 (12.3%)
            text_labels = abs_vals.index.map(lambda idx: f"{abs_vals[idx]:,} ({pct_vals[idx]})") if len(abs_vals) > 0 else []
            fig2.add_trace(go.Bar(name=seg, x=pivot.index, y=pivot[seg], marker_color=colors.get(seg),
                                  text=[f"{abs_vals[i]:,} ({pct_vals[i]})" for i in range(len(abs_vals))], 
                                  textposition='inside'))
        fig2.update_layout(barmode='stack', title='Qualified / Open / Disqualified by Priority (stacked)', height=420, legend=dict(title='Segment'))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        # Fallback: no status column — show Qualified vs Other stacked
        agg['other_leads'] = agg['total_leads'] - agg['qualified_leads']
        fig2 = go.Figure()
        q_vals = agg['qualified_leads'].astype(float).fillna(0)
        o_vals = agg['other_leads'].astype(float).fillna(0)
        row_total = agg['total_leads'].astype(float).fillna(0)
        # Calculate percentages
        q_pct = (q_vals / row_total * 100).fillna(0)
        o_pct = (o_vals / row_total * 100).fillna(0)
        # Format labels as: 1,234 (12.3%)
        q_labels = [f"{int(round(q_vals[i])):,} ({q_pct[i]:.1f}%)" for i in range(len(q_vals))]
        o_labels = [f"{int(round(o_vals[i])):,} ({o_pct[i]:.1f}%)" for i in range(len(o_vals))]
        fig2.add_trace(go.Bar(name='Qualified', x=agg.index, y=agg['qualified_leads'], marker_color='green', text=q_labels, textposition='inside'))
        fig2.add_trace(go.Bar(name='Other', x=agg.index, y=agg['other_leads'], marker_color='lightgray', text=o_labels, textposition='inside'))
        fig2.update_layout(barmode='stack', title='Qualified vs Other by Priority (stacked)', height=400, legend=dict(title='Segment'))
        st.plotly_chart(fig2, use_container_width=True)

st.markdown('### Conversion % Over Time')
if 'yyyymm' in filtered.columns and not filtered.empty:
    if hp_col:
        grp = filtered.groupby(['yyyymm','_priority_norm'])[hp_col].sum().rename('total_leads').reset_index()
        qual = filtered.loc[filtered['_is_qualified']].groupby(['yyyymm','_priority_norm'])[hp_col].sum().rename('qualified_leads').reset_index()
        grp = grp.merge(qual, on=['yyyymm','_priority_norm'], how='left').fillna(0)
    else:
        grp = filtered.groupby(['yyyymm','_priority_norm']).agg(total_leads=('priority','size'), qualified_leads=('_is_qualified','sum')).reset_index()
    grp['conversion_pct'] = 100 * grp['qualified_leads'] / grp['total_leads'].replace({0: np.nan})
    pivot = grp.pivot(index='yyyymm', columns='_priority_norm', values='conversion_pct').fillna(0).reset_index().sort_values('yyyymm')
    fig_ts = go.Figure()
    for pr in ['Urgent','Normal']:
        if pr in pivot.columns:
            # add labels on markers for readability
            fig_ts.add_trace(go.Scatter(x=pivot['yyyymm'], y=pivot[pr], mode='lines+markers+text', name=pr,
                                        text=pivot[pr].round(1).map(lambda v: f"{v:.1f}%"), textposition='top center'))
    fig_ts.update_layout(title='Conversion % Over Time', xaxis_title='Month (YYYY-MM)', yaxis_title='Conversion %', height=480)
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info('No yyyymm column or no data for time series.')

st.markdown('### Average TOV Analysis')
if 'tov' in filtered.columns and pd.api.types.is_numeric_dtype(filtered['tov']):
    avg_tov_overall = filtered['tov'].mean()
    st.write(f"Average TOV (overall): {int(round(avg_tov_overall)):,}")

    # TOV by priority
    tov_by_pr = filtered.groupby('_priority_norm')['tov'].mean().reindex(['Urgent','Normal']).fillna(0)
    tov_by_pr_display = tov_by_pr.map(lambda v: int(round(v)))
    fig_tov_pr = px.bar(tov_by_pr.reset_index(), x='_priority_norm', y='tov', labels={'_priority_norm':'Priority','tov':'Average TOV'}, text=tov_by_pr_display.map(lambda v: f"{v:,}"))
    fig_tov_pr.update_layout(title='Average TOV by Priority', height=350)
    st.plotly_chart(fig_tov_pr, use_container_width=True)

    # TOV over time by priority
    if 'yyyymm' in filtered.columns:
        tov_ts = filtered.groupby(['yyyymm','_priority_norm'])['tov'].mean().reset_index()
        pivot_tov = tov_ts.pivot(index='yyyymm', columns='_priority_norm', values='tov').fillna(0).reset_index().sort_values('yyyymm')
        fig_tov_ts = go.Figure()
        for pr in ['Urgent','Normal']:
            if pr in pivot_tov.columns:
                # show integer monetary-like labels without decimals
                text_vals = pivot_tov[pr].fillna(0).map(lambda v: f"{int(round(v)):,}")
                fig_tov_ts.add_trace(go.Scatter(x=pivot_tov['yyyymm'], y=pivot_tov[pr], mode='lines+markers+text', name=pr,
                                                text=text_vals, textposition='top center'))
        fig_tov_ts.update_layout(title='Average TOV Over Time', xaxis_title='Month (YYYY-MM)', yaxis_title='Average TOV', height=420)
        st.plotly_chart(fig_tov_ts, use_container_width=True)
else:
    st.info('TOV column not available or not numeric.')

st.markdown('### Filtered sample (first 200 rows)')
st.dataframe(filtered.head(200))

csv_bytes = df_to_bytes(filtered)
st.download_button('Download filtered CSV', data=csv_bytes, file_name='filtered_leads.csv', mime='text/csv')

st.markdown('---')

# --- Enhanced multi-perspective dashboard (tabs) ---
st.header('Enhanced: Multi-Perspective Conversion Analysis')
tab_labels = ['By Market','By Country','By Business Group','By Campaign Type','Heatmap (Market × Priority)','Summary Table']
tabs = st.tabs(tab_labels)

def conversion_by_dim(filtered_df, dim_col, top_n=None):
    if dim_col not in filtered_df.columns:
        return pd.DataFrame()
    if hp_col:
        agg = filtered_df.groupby(dim_col)[hp_col].agg(total_leads='sum')
        agg['qualified_leads'] = filtered_df.loc[filtered_df['_is_qualified']].groupby(dim_col)[hp_col].sum()
    else:
        agg = filtered_df.groupby(dim_col).size().rename('total_leads').to_frame()
        agg['qualified_leads'] = filtered_df.loc[filtered_df['_is_qualified']].groupby(dim_col).size()
    agg = agg.fillna(0)
    agg['conversion_pct'] = 100 * agg['qualified_leads'] / agg['total_leads'].replace({0: np.nan})
    res = agg.sort_values('conversion_pct', ascending=False).reset_index()
    if top_n:
        return res.head(top_n)
    return res

# Tab 0: By Market
with tabs[0]:
    st.subheader('Conversion % by Market')
    if 'market' in filtered.columns:
        by_market = conversion_by_dim(filtered, 'market')
        if not by_market.empty:
            fig = px.bar(by_market, x='market', y='conversion_pct', color='conversion_pct', text=by_market['conversion_pct'].map(lambda v: f"{v:.1f}%"),
                         color_continuous_scale='RdYlGn', labels={'conversion_pct':'Conversion %'})
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            display_df = by_market.copy()
            if 'total_leads' in display_df.columns:
                display_df['total_leads'] = display_df['total_leads'].map(lambda v: f"{int(round(v)):,}")
            if 'qualified_leads' in display_df.columns:
                display_df['qualified_leads'] = display_df['qualified_leads'].map(lambda v: f"{int(round(v)):,}")
            display_df['conversion_pct'] = display_df['conversion_pct'].map(lambda v: f"{v:.1f}%")
            st.dataframe(display_df)
        else:
            st.info('No market data available.')
    else:
        st.info('Market column not available in dataset.')

# Tab 1: By Country
with tabs[1]:
    st.subheader('Conversion % by Country (Top 15)')
    if 'country' in filtered.columns:
        by_country = conversion_by_dim(filtered, 'country')
        if not by_country.empty:
            fig = px.bar(by_country.head(15), x='country', y='conversion_pct', color='conversion_pct', text=by_country.head(15)['conversion_pct'].map(lambda v: f"{v:.1f}%"),
                         color_continuous_scale='RdYlGn', labels={'conversion_pct':'Conversion %'})
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            display_df = by_country.head(15).copy()
            if 'total_leads' in display_df.columns:
                display_df['total_leads'] = display_df['total_leads'].map(lambda v: f"{int(round(v)):,}")
            if 'qualified_leads' in display_df.columns:
                display_df['qualified_leads'] = display_df['qualified_leads'].map(lambda v: f"{int(round(v)):,}")
            display_df['conversion_pct'] = display_df['conversion_pct'].map(lambda v: f"{v:.1f}%")
            st.dataframe(display_df)
        else:
            st.info('No country data available.')
    else:
        st.info('Country column not available in dataset.')

# Tab 2: By Business Group
with tabs[2]:
    st.subheader('Conversion % by Business Group')
    if 'business_group' in filtered.columns:
        by_bg = conversion_by_dim(filtered, 'business_group')
        if not by_bg.empty:
            fig = px.bar(by_bg, x='business_group', y='conversion_pct', color='conversion_pct', text=by_bg['conversion_pct'].map(lambda v: f"{v:.1f}%"),
                         color_continuous_scale='RdYlGn', labels={'conversion_pct':'Conversion %'})
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            display_df = by_bg.copy()
            if 'total_leads' in display_df.columns:
                display_df['total_leads'] = display_df['total_leads'].map(lambda v: f"{int(round(v)):,}")
            if 'qualified_leads' in display_df.columns:
                display_df['qualified_leads'] = display_df['qualified_leads'].map(lambda v: f"{int(round(v)):,}")
            display_df['conversion_pct'] = display_df['conversion_pct'].map(lambda v: f"{v:.1f}%")
            st.dataframe(display_df)
        else:
            st.info('No business group data available.')
    else:
        st.info('Business Group column not available in dataset.')

# Tab 3: By Campaign Type (replacing Job Title)
with tabs[3]:
    st.subheader('Conversion % by Campaign Type (Top 15)')
    if 'campaign_type' in filtered.columns:
        by_ct = conversion_by_dim(filtered, 'campaign_type')
        if not by_ct.empty:
            fig = px.bar(by_ct.head(15), x='campaign_type', y='conversion_pct', color='conversion_pct', text=by_ct.head(15)['conversion_pct'].map(lambda v: f"{v:.1f}%"),
                         color_continuous_scale='RdYlGn')
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            display_df = by_ct.head(15).copy()
            if 'total_leads' in display_df.columns:
                display_df['total_leads'] = display_df['total_leads'].map(lambda v: f"{int(round(v)):,}")
            if 'qualified_leads' in display_df.columns:
                display_df['qualified_leads'] = display_df['qualified_leads'].map(lambda v: f"{int(round(v)):,}")
            display_df['conversion_pct'] = display_df['conversion_pct'].map(lambda v: f"{v:.1f}%")
            st.dataframe(display_df)
        else:
            st.info('No campaign_type data available.')
    else:
        st.info('Campaign Type column not available in dataset.')

# Tab 4: Heatmap Market x Priority
with tabs[4]:
    st.subheader('Heatmap: Conversion % by Market × Priority')
    if 'market' in filtered.columns and '_priority_norm' in filtered.columns:
        if hp_col:
            pivot_data = filtered.groupby(['market','_priority_norm'])[hp_col].agg(total_leads='sum').reset_index()
            try:
                pivot_data['qualified_leads'] = filtered.loc[filtered['_is_qualified']].groupby(['market','_priority_norm'])[hp_col].sum().values
            except Exception:
                qual = filtered.loc[filtered['_is_qualified']].groupby(['market','_priority_norm'])[hp_col].sum().reset_index().rename(columns={hp_col:'qualified_leads'})
                pivot_data = pivot_data.merge(qual, on=['market','_priority_norm'], how='left')
        else:
            pivot_data = filtered.groupby(['market','_priority_norm']).agg(total_leads=('priority','size')).reset_index()
            try:
                pivot_data['qualified_leads'] = filtered.loc[filtered['_is_qualified']].groupby(['market','_priority_norm']).size().values
            except Exception:
                qual = filtered.loc[filtered['_is_qualified']].groupby(['market','_priority_norm']).size().reset_index().rename(columns={0:'qualified_leads'})
                pivot_data = pivot_data.merge(qual, on=['market','_priority_norm'], how='left')
        pivot_data = pivot_data.fillna(0)
        pivot_data['conversion_pct'] = 100 * pivot_data['qualified_leads'] / pivot_data['total_leads'].replace({0: np.nan})
        heatmap_pivot = pivot_data.pivot(index='market', columns='_priority_norm', values='conversion_pct').fillna(0)
        fig = px.imshow(heatmap_pivot, labels=dict(x='Priority', y='Market', color='Conversion %'),
                       color_continuous_scale='RdYlGn', text_auto='.1f', zmin=0, zmax=100, aspect='auto')
        fig.update_layout(height=max(300, heatmap_pivot.shape[0]*20))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(heatmap_pivot.round(2))
    else:
        st.info('Market or Priority column not available for heatmap.')

# Tab 5: Summary Table
with tabs[5]:
    st.subheader('Segment Performance Summary')
    st.write(f"Total filtered leads: {len(filtered):,}")
    st.write(f"Qualified: {filtered['_is_qualified'].sum():,}")
    overall_conv = (filtered['_is_qualified'].sum() / len(filtered) * 100) if len(filtered) else 0
    st.write(f"Overall conversion: {overall_conv:.1f}%")
    if 'country' in filtered.columns:
        top_countries = conversion_by_dim(filtered, 'country').head(5)
        bottom_countries = conversion_by_dim(filtered, 'country').tail(5)
        st.markdown('**Top 5 Countries by Conversion %**')
        disp_top = top_countries.copy()
        if 'total_leads' in disp_top.columns:
            disp_top['total_leads'] = disp_top['total_leads'].map(lambda v: f"{int(round(v)):,}")
        if 'qualified_leads' in disp_top.columns:
            disp_top['qualified_leads'] = disp_top['qualified_leads'].map(lambda v: f"{int(round(v)):,}")
        disp_top['conversion_pct'] = disp_top['conversion_pct'].map(lambda v: f"{v:.1f}%")
        st.dataframe(disp_top)
        st.markdown('**Bottom 5 Countries by Conversion %**')
        disp_bottom = bottom_countries.copy()
        if 'total_leads' in disp_bottom.columns:
            disp_bottom['total_leads'] = disp_bottom['total_leads'].map(lambda v: f"{int(round(v)):,}")
        if 'qualified_leads' in disp_bottom.columns:
            disp_bottom['qualified_leads'] = disp_bottom['qualified_leads'].map(lambda v: f"{int(round(v)):,}")
        disp_bottom['conversion_pct'] = disp_bottom['conversion_pct'].map(lambda v: f"{v:.1f}%")
        st.dataframe(disp_bottom)
    if 'business_group' in filtered.columns:
        st.markdown('**Top 5 Business Groups by Conversion %**')
        bg_top = conversion_by_dim(filtered, 'business_group').head(5).copy()
        if 'total_leads' in bg_top.columns:
            bg_top['total_leads'] = bg_top['total_leads'].map(lambda v: f"{int(round(v)):,}")
        if 'qualified_leads' in bg_top.columns:
            bg_top['qualified_leads'] = bg_top['qualified_leads'].map(lambda v: f"{int(round(v)):,}")
        bg_top['conversion_pct'] = bg_top['conversion_pct'].map(lambda v: f"{v:.1f}%")
        st.dataframe(bg_top)

st.markdown('---')
st.caption('Notes: Priority is normalized to Urgent vs Normal. If your dataset has different column names, update code to match them.')
