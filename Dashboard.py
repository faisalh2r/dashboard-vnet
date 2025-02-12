import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns 
from babel.numbers import format_currency        
from io import BytesIO  # Impor BytesIO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from random import sample
import numpy as np
from numpy.random import uniform
from math import isnan
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from streamlit_extras.metric_cards import style_metric_cards
import time

st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide") 

theme_plotly = None 
# load Style css
with open('css/style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#-----Fungsi Dataset Income
def create_daily_incomes_df(df):
    daily_incomes_df = df.resample(rule='D', on='Tanggal_Aktif').agg({
        'Invoice': 'nunique',
        'Pendapatan_Bersih': 'sum'
    })
    daily_incomes_df = daily_incomes_df.reset_index()
    daily_incomes_df.rename(columns={
        'Invoice': 'income_count',
        'Pendapatan_Bersih': 'revenue'
    }, inplace=True)

    return daily_incomes_df

def create_daily_incomes_ppn_df(df):
    daily_incomes_ppn_df = df.resample(rule='D', on='Tanggal_Aktif').agg({
        'Invoice': 'nunique',
        'Total': 'sum'
    })
    daily_incomes_ppn_df = daily_incomes_ppn_df.reset_index()
    daily_incomes_ppn_df.rename(columns={
        'Invoice': 'income_count',
        'Total': 'revenue'
    }, inplace=True)

    return daily_incomes_ppn_df
 
def create_daily_fee_seller_df(df):
    daily_fee_seller_df = df.resample(rule='D', on='Tanggal_Aktif').agg({
        'Invoice': 'nunique',
        'Fee_Seller': 'sum'
    })
    daily_fee_seller_df = daily_fee_seller_df.reset_index()
    daily_fee_seller_df.rename(columns={
        'Invoice': 'income_count',
        'Fee_Seller': 'pengeluaran'
    }, inplace=True)

    return daily_fee_seller_df
 
def create_sum_order_items2_df(df):
    paket_counts = df.groupby('Paket_Langganan')['Invoice'].nunique()
    # Mengurutkan hasil berdasarkan jumlah pemasangan secara menurun
    paket_counts = paket_counts.sort_values(ascending=False)
    # Mengubah menjadi DataFrame
    sum_order_items2_df = paket_counts.reset_index()
    # Mengganti nama kolom
    sum_order_items2_df.columns = ['Paket_Langganan', 'quantity_x']
    
    return sum_order_items2_df

def create_byownerdata_df(df):
    byownerdata_df = df.groupby(by="Owner_Data").Invoice.nunique().reset_index()
    byownerdata_df.rename(columns={
        "Invoice": "customer_count"
    }, inplace=True)
    
    return byownerdata_df 

def create_scatter_income_vs_duration(df):
    scatter_df = df[['Lama_Berlangganan(Bulan)', 'Pendapatan_Bersih']]
    correlation = scatter_df['Lama_Berlangganan(Bulan)'].corr(scatter_df['Pendapatan_Bersih'])

    return scatter_df, correlation
           
def create_customers_by_duration_area(df):
    # Menghitung jumlah pelanggan per area dan lama berlangganan
    grouped_df = df.groupby(['Lama_Berlangganan(Bulan)', 'Area']).size().reset_index(name='customer_count')
    # Menghitung rata-rata jumlah pelanggan per area
    avg_customers_per_area = grouped_df.groupby('Area')['customer_count'].mean().reset_index()
    avg_customers_per_area.rename(columns={'customer_count': 'average_customers_per_area'}, inplace=True)
    
    return grouped_df, avg_customers_per_area    

#Fungsi untuk membuat RFM DataFrame
def create_rfm2_df(df):
    rfm2_df = df.groupby(by='ID_Pelanggan', as_index=False).agg({
        "Tanggal_Aktif": "max",  
        "Invoice": "nunique",  
        "Total": "sum"
    })
    
    rfm2_df.columns = ['ID_Pelanggan', 'max_order_timestamp', 'frequency', 'monetary']
    rfm2_df["max_order_timestamp"] = rfm2_df["max_order_timestamp"].dt.date
    recent_date = df["Tanggal_Aktif"].dt.date.max()
    rfm2_df["recency"] = rfm2_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm2_df.drop("max_order_timestamp", axis=1, inplace=True)
    
    return rfm2_df

def check_outliers_rfm_streamlit(rfm2_df):
    num_features = rfm2_df[['recency', 'frequency', 'monetary']]

    r, c = 0, 0
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))  # Membuat grid 2x2 untuk boxplot

    for n, i in enumerate(num_features.columns):
        sns.boxplot(x=i, data=num_features, ax=ax[r, c])  # Membuat boxplot
        c += 1
        if (n + 1) % 2 == 0:  # Berpindah baris setelah 2 kolom
            r += 1
            c = 0
    
    ax[r, c].axis("off")  # Mematikan grid kosong jika ada
    plt.tight_layout()  # Menata tata letak plot

    st.pyplot(fig)  # Menampilkan plot di Streamlit

# Fungsi untuk melakukan capping outlier pada RFM
def cap_outliers_rfm_streamlit(rfm2_df, lower_cap=0.05, upper_cap=0.95):
    # Capping untuk monetary
    h_cap_monetary = rfm2_df['monetary'].quantile(upper_cap)
    l_cap_monetary = rfm2_df['monetary'].quantile(lower_cap)
    rfm2_df['monetary'] = rfm2_df['monetary'].clip(lower=l_cap_monetary, upper=h_cap_monetary)
    
    # Capping untuk recency
    h_cap_recency = rfm2_df['recency'].quantile(upper_cap)
    rfm2_df['recency'] = rfm2_df['recency'].clip(upper=h_cap_recency)

    return rfm2_df

# Fungsi untuk menghitung Hopkins Statistic
# mengevaluasi apakah data memiliki kecenderungan untuk membentuk kluster atau tidak. Nilai yang lebih tinggi menunjukkan bahwa data lebih terkluster daripada acak
def hopkins(X):
    d = X.shape[1]  # jumlah dimensi
    n = len(X)  # jumlah data
    m = int(0.1 * n)  # sampel 10% dari data
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)  # nearest neighbors

    # Random sampel dari data dan dari distribusi uniform
    rand_X = sample(range(0, n, 1), m)

    ujd = []  # jarak untuk sampel dari distribusi uniform
    wjd = []  # jarak untuk sampel dari data asli
    for j in range(0, m):
        # menghitung jarak dari distribusi uniform
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X, axis=0), d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        
        # menghitung jarak dari data asli
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    # Menghitung Hopkins Statistic
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0

    return H

#--------Fungsi Dataset Pelanggan 2        
def create_daily_installations_df(df):
    
    df_paid = df[df['Status'] == 'PAID']

    daily_installations_df = df_paid.resample(rule='D', on='Installation_Date').agg({
        'Registration_Number': 'nunique',
        'Final_Amount': 'sum'
    })
    daily_installations_df = daily_installations_df.reset_index()
    daily_installations_df.rename(columns={
        'Registration_Number': 'install_count',
        'Final_Amount': 'revenue'
    }, inplace=True)
    
    # Menambahkan kolom baru untuk 35% dan 65% dari revenue
    daily_installations_df['revenue_35_percent'] = daily_installations_df['revenue'] * 0.35
    daily_installations_df['revenue_65_percent'] = daily_installations_df['revenue'] * 0.65
    
    return daily_installations_df

def create_daily_installations2_df(df):
    
    df_unpaid = df[df['Status'] == 'UNPAID']

    daily_installations2_df = df_unpaid.resample(rule='D', on='Installation_Date').agg({
        'Registration_Number': 'nunique',
        'Final_Amount': 'sum'
    })
    daily_installations2_df = daily_installations2_df.reset_index()
    daily_installations2_df.rename(columns={
        'Registration_Number': 'install_count',
        'Final_Amount': 'revenue'
    }, inplace=True)
    
    # Menambahkan kolom baru untuk 35% dan 65% dari revenue
    daily_installations2_df['revenue_35_percent'] = daily_installations2_df['revenue'] * 0.35
    daily_installations2_df['revenue_65_percent'] = daily_installations2_df['revenue'] * 0.65
    
    return daily_installations2_df

def create_daily_registration_df(df):
    # Mengelompokkan data berdasarkan 'Installation_Date' dan 'Type'
    daily_registration_df = df.groupby([df['Installation_Date'].dt.date, 'Type','Status']).agg({
        'Registration_Number': 'nunique',
        'Final_Amount': 'sum'
    }).reset_index()

    # Mengubah nama kolom
    daily_registration_df.rename(columns={
        'Registration_Number': 'install_count',
        'Status': 'description',
        'Final_Amount': 'revenue'
    }, inplace=True)
    
    return daily_registration_df
    
def create_sum_order_items_df(df):
    paket_counts = df.groupby('Internet_Package')['Registration_Number'].nunique()
    # Mengurutkan hasil berdasarkan jumlah pemasangan secara menurun
    paket_counts = paket_counts.sort_values(ascending=False)
    # Mengubah menjadi DataFrame
    sum_order_items_df = paket_counts.reset_index()
    # Mengganti nama kolom
    sum_order_items_df.columns = ['Internet_Package', 'quantity_x']
    
    return sum_order_items_df

def create_bystatus_df(df):
    bystatus_df = df.groupby(by='Status_Customer').Registration_Number.nunique().reset_index()
    bystatus_df.rename(columns={
        'Registration_Number': 'customer_count'
    }, inplace=True)

    return bystatus_df

def create_bytype_df(df):
    bytype_df =  df.groupby(by='Type').Registration_Number.nunique().reset_index()
    bytype_df.rename(columns={
        'Registration_Number': 'customer_count'
    }, inplace=True)

    return bytype_df

def create_byownership_status_df(df):
    byownership_df = df.groupby(by='Building_Ownership_Status').Registration_Number.nunique().reset_index()
    byownership_df.rename(columns={
        'Registration_Number': 'customer_count'
    }, inplace=True)

    return byownership_df

def create_bystatus_paid_df(df):
    bystatus_paid_df = df.groupby(by='Status').Registration_Number.nunique().reset_index()
    bystatus_paid_df.rename(columns={
        'Registration_Number': 'customer_count'
    }, inplace=True)

    return bystatus_paid_df

def create_bypayment_method_df(df):
    bypayment_method_df = df.groupby(by='Payment_Method').Registration_Number.nunique().reset_index()
    bypayment_method_df.rename(columns={
        'Registration_Number': 'customer_count'
    }, inplace=True)

    return bypayment_method_df
    
def create_rfm_df(df):
    rfm_df = df.groupby(by='Customer_Number', as_index=False).agg({
        "Installation_Date": "max", #mengambil tanggal order terakhir
        "Billing_Number": "nunique",
        "Final_Amount": "sum"
    })
    rfm_df.columns = ['Customer_Number', 'max_order_timestamp', 'frequency','monetary']
    
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = df["Installation_Date"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
    
    return rfm_df

# Fungsi untuk mengonversi DataFrame ke format Excel
def convert_df_to_excel(df):
    # Membuat buffer
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()
 
st.header('🌐Dashboard Pelanggan Internet V-NET Indonesia') 

st.sidebar.image("images/logo.png",caption="")

tab1, tab2, = st.tabs(['📊Data Pelanggan VNET dan Viberlink','📊Data Pelanggan VNET (Jakarta,Bandung,Sukabumi)']) 
 
with tab1:
    with st.expander("Upload Dataset Pelanggan", expanded=True, icon="📤"):
        dataset = st.file_uploader(label = 'Upload your dataset pelanggan : ', type=['xlsx'], label_visibility='hidden')
    if dataset is not None:
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()
        try:
            with st.sidebar:
                all_df = pd.read_excel(dataset)
                datetime_columns = ["Installation_Date", "Due_Date"]
                all_df.sort_values(by="Installation_Date", inplace=True)
                all_df.reset_index(inplace=True)
                
                for column in datetime_columns:
                    all_df[column] = pd.to_datetime(all_df[column])
                    
                min_date = all_df["Installation_Date"].min()
                max_date = all_df["Installation_Date"].max()
                owners = all_df['Owner'].unique()  # Mengambil daftar area unik

                    
                    # Mengambil start_date & end_date dari date_input
                start_date, end_date = st.date_input(
                    label='📅 Rentang Waktu Pelanggan Viberlink dan VNET',min_value=min_date,
                    max_value=max_date,
                    value=[min_date, max_date]
                    )
                
                selected_owner = st.selectbox('🌍Pilih Owner', options=owners, index=0) # owner

            
            main_df = all_df[(all_df["Installation_Date"] >= str(start_date)) & 
                            # (all_df["Installation_Date"] <= str(end_date  + pd.Timedelta(days=1))) &\
                            (all_df["Installation_Date"] <= str(end_date)) &
                            (all_df['Owner'] == selected_owner)
                            ]
                
            sum_order_items_df = create_sum_order_items_df(main_df)
            daily_installations_df = create_daily_installations_df(main_df)
            daily_installations2_df = create_daily_installations2_df(main_df)
            bystatus_df = create_bystatus_df(main_df)
            bytype_df = create_bytype_df(main_df)
            daily_registration_df = create_daily_registration_df(main_df)
            byownership_df = create_byownership_status_df(main_df)
            bystatus_paid_df = create_bystatus_paid_df(main_df)
            bypayment_method_df = create_bypayment_method_df(main_df)
            # bychurn_df = create_bychurn_df(main_df)
            rfm_df = create_rfm_df(main_df)    
            # churn_percentage_df, overall_churn_percentage = create_churn_percentage_df(main_df)
            
            st.subheader('Daily Installations')
            # Membuat visualisasi dengan Plotly
            # graf = st.columns(1, gap='small')
            # with graf:
          
            rev, rev2, = st.columns(2, gap='small')
            col1, col2, col3, col4, = st.columns(4)
            if selected_owner == 'VNET':
                with rev:
                    st.info('Total Revenue',icon="💰")
                    total_paid_revenue = format_currency(daily_installations_df.revenue.sum(), "IDR", locale='id_ID') 
                    st.metric("Total", value=total_paid_revenue)
                with rev2:
                    st.info('Total Trade Receivables',icon="💰")
                    total_unpaid_revenue = format_currency(daily_installations2_df.revenue.sum(), "IDR", locale='id_ID') 
                    st.metric("Total", value=total_unpaid_revenue)
            if selected_owner == 'Viberlink':
                with col1:
                    st.info('Total Revenue (PAID)',icon="💰")
                    total_paid_revenue = format_currency(daily_installations_df.revenue.sum(), "IDR", locale='id_ID') 
                    st.metric("Total", value=total_paid_revenue)
                with col2:
                    st.info('Total Trade Receivables (UNPAID)',icon="💰")
                    total_unpaid_revenue = format_currency(daily_installations2_df.revenue.sum(), "IDR", locale='id_ID') 
                    st.metric("Total", value=total_unpaid_revenue)
                with col3:
                    st.info('Total Revenue Viberlink 65%',icon="💰")
                    total_revenue_65_percent = format_currency(daily_installations_df['revenue_65_percent'].sum(), "IDR", locale='id_ID')
                    st.metric("Total Revenue", value=total_revenue_65_percent)
                with col4:
                    st.info('Total Revenue VNET 35%',icon="💰")
                    total_revenue_35_percent = format_currency(daily_installations_df['revenue_35_percent'].sum(), "IDR", locale='id_ID')
                    st.metric("Total Revenue", value=total_revenue_35_percent)                           
                  
            style_metric_cards(background_color="#FFFFFF",border_left_color="#29abe2",border_color="#000000",box_shadow="#F71938")   
                            
            trans1, trans2, = st.columns(2)
           
            with trans1:
                fig = go.Figure()
                
                registration_df = daily_registration_df[daily_registration_df['Type'] == 'REGISTRATION']

                # Menambahkan data pemasangan
                fig.add_trace(go.Scatter(
                    x=registration_df["Installation_Date"],
                    y=registration_df["install_count"],
                    mode='lines+markers',
                    marker=dict(color='#0000FF'),
                    line=dict(width=2),
                    name='Registration',
                    hovertemplate="<b>Tanggal: %{x}<br>Total Transaksi: %{y}<br>Pendapatan: Rp %{customdata:,.0f}<extra></extra></b>",
                    customdata=registration_df['revenue']
                ))

                # Mengatur layout
                fig.update_layout(
                    title="<b>📈 Total Transaksi Harian</b>",
                    xaxis_title='Tanggal',
                    yaxis_title='Total Transaksi',
                    template="plotly_white",
                    xaxis_tickangle=-45,
                    yaxis=dict(title='Total Transaksi', side='left', showgrid=True),
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    height=600,
                    width=900
                )
                
                # customer_df = daily_registration_df[daily_registration_df['Type'] == 'CUSTOMER']
                customer_df = daily_registration_df[(daily_registration_df['Type'] == 'CUSTOMER') & 
                                            (daily_registration_df['description'] == 'PAID')]  
                
                customer_df['customdata'] = customer_df.apply(
                    lambda row: f"Rp {row['revenue']:,} {row['description']} = {row['install_count']}", axis=1
                )
                
                    # Menambahkan data pemasangan
                fig.add_trace(go.Scatter(
                    x=customer_df["Installation_Date"],
                    y=customer_df["install_count"],
                    mode='lines+markers',
                    marker=dict(color='#008000'),
                    line=dict(width=2),
                    name='Customer (PAID)',
                    hovertemplate="<b>Tanggal: %{x}<br>Total Transaksi: %{y}<br>Pendapatan:%{customdata}<extra></extra></b>",
                    customdata=customer_df['customdata']
                ))

                # Mengatur layout
                fig.update_layout(
                    title="<b>📈 Total Transaksi Harian</b>",
                    xaxis_title='Tanggal',
                    yaxis_title='Total Transaksi',
                    template="plotly_white",
                    xaxis_tickangle=-45,
                    yaxis=dict(title='Total Transaksi', side='left', showgrid=True),
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    height=600,
                    width=900
                )

                customer_df = daily_registration_df[(daily_registration_df['Type'] == 'CUSTOMER') & 
                                            (daily_registration_df['description'] == 'UNPAID')]  
                
                customer_df['customdata'] = customer_df.apply(
                    lambda row: f"Rp {row['revenue']:,} {row['description']} = {row['install_count']}", axis=1
                )
                
                # Menambahkan data pemasangan
                fig.add_trace(go.Scatter(
                    x=customer_df["Installation_Date"],
                    y=customer_df["install_count"],
                    mode='lines+markers',
                    marker=dict(color='#ffde59'),
                    line=dict(width=2),
                    name='Customer (UNPAID)',
                    hovertemplate="<b>Tanggal: %{x}<br>Total Transaksi: %{y}<br>Pendapatan:%{customdata}<extra></extra></b>",
                    customdata=customer_df['customdata']
                ))

                # Mengatur layout
                fig.update_layout(
                    title="<b>📈 Total Transaksi Harian</b>",
                    xaxis_title='Tanggal',
                    yaxis_title='Total Transaksi',
                    template="plotly_white",
                    xaxis_tickangle=-45,
                    yaxis=dict(title='Total Transaksi', side='left', showgrid=True),
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    height=600,
                    width=900
                )


                # Menampilkan grafik total pemasangan
                st.plotly_chart(fig, use_container_width=True) 
            
            with trans2:
                # Filter untuk kategori REGISTRATION
                registration_df = daily_registration_df[daily_registration_df['Type'] == 'REGISTRATION']
                registration_count = registration_df['install_count'].sum()

                # Filter untuk kategori CUSTOMER PAID
                customer_paid_df = daily_registration_df[(daily_registration_df['Type'] == 'CUSTOMER') & 
                                                        (daily_registration_df['description'] == 'PAID')]
                customer_paid_count = customer_paid_df['install_count'].sum()

                # Filter untuk kategori CUSTOMER UNPAID
                customer_unpaid_df = daily_registration_df[(daily_registration_df['Type'] == 'CUSTOMER') & 
                                                            (daily_registration_df['description'] == 'UNPAID')]
                customer_unpaid_count = customer_unpaid_df['install_count'].sum()

                # Data untuk Pie chart
                labels = ['Registration', 'Customer (Paid)', 'Customer (Unpaid)']
                values = [registration_count, customer_paid_count, customer_unpaid_count]

                # Membuat Pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=labels, 
                    values=values, 
                    hole=0.3,  # Membuat donut chart dengan hole di tengah
                    hovertemplate="<b>%{label}</b><br>Total Transaksi: %{value}<extra></extra>", # Menampilkan total transaksi dan pendapatan pada hover
                    textinfo='percent+label',  # Menampilkan persen dan label
                    marker=dict(colors=['#0000FF', '#008000', '#FFDE59'])  # Warna untuk masing-masing kategori
                )])

                # Menambahkan judul dan layout untuk Pie chart
                fig.update_layout(
                    title="<b>📊 Proporsi Transaksi Berdasarkan Kategori</b>",
                    template="plotly_white",
                    height=600,
                    width=700,
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)'
                )

                # Menampilkan grafik Pie chart
                st.plotly_chart(fig, use_container_width=True)
                
                total_status_active = bystatus_df[bystatus_df['Status_Customer'] == 'Active']['customer_count'].sum()
                st.markdown(f"- **Total Customer Active: {total_status_active}**")
                
    
            col_best, col_worst, col_type, col_pay, = st.columns(4, gap='small')
            
            best_products = sum_order_items_df.head(5)
            worst_products = sum_order_items_df.sort_values(by="quantity_x", ascending=True).head(5)
            with col_best:
                # Membuat figure untuk Best Performing Product
                fig_best = px.bar(
                    best_products,
                    x="quantity_x",
                    y="Internet_Package",
                    orientation='h',
                    title="<b>📈 Best Performing Product</b>",
                    color_discrete_sequence=["#90CAF9"] * len(best_products),
                    labels={"quantity_x": "Number of Sales", "Internet_Package": "Paket"},
                    template="plotly_white"
                )

                fig_best.update_layout(
                    xaxis=dict(title="Number of Sales", showgrid=True, gridcolor='#cecdcd'),
                    yaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    height=400

                )

                # Menampilkan grafik produk terbaik
                st.plotly_chart(fig_best, use_container_width=True)
            with col_worst:
            # Membuat figure untuk Worst Performing Product
                fig_worst = px.bar(
                    worst_products,
                    x="quantity_x",
                    y="Internet_Package",
                    orientation='h',
                    title="<b>📉 Worst Performing Product</b>",
                    color_discrete_sequence=["#90CAF9"] * len(worst_products),
                    labels={"quantity_x": "Number of Sales", "Internet_Package": "Paket"},
                    template="plotly_white"
                )

                fig_worst.update_layout(
                    xaxis=dict(title="Number of Sales", showgrid=True, gridcolor='#cecdcd'),
                    yaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    yaxis_tickmode='linear',
                    yaxis_tickvals=list(range(len(worst_products))),
                    height=400

                )

                # Menampilkan grafik produk terburuk
                st.plotly_chart(fig_worst, use_container_width=True)

            with col_type:
                # Visualisasi untuk jumlah pelanggan berdasarkan tipe
                fig_type = px.bar(
                    bytype_df.sort_values(by="Type", ascending=False),
                    x="Type",
                    y="customer_count",
                    title="<b>👤Number of Customer by Type</b>",
                    color_discrete_sequence=["#90CAF9"] * len(bytype_df),
                    labels={"customer_count": "Number of Customers", "Type": "Tipe"},
                    template="plotly_white"
                )

                fig_type.update_layout(
                    xaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                    yaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                    yaxis_tickangle=0,
                    height=400
                )

                # Menampilkan grafik jumlah pelanggan berdasarkan status
                st.plotly_chart(fig_type, use_container_width=True)
                
            with col_pay:
                fig_pay = px.bar(
                    bystatus_paid_df.sort_values(by="Status", ascending=False),
                    x="Status",
                    y="customer_count",
                    title="<b>👤 Number of Customer by Payment Status</b>",
                    color_discrete_sequence=["#90CAF9"] * len(bystatus_paid_df),
                    labels={"customer_count": "Number of Customers", "Status": "Status Pembayaran"},
                    template="plotly_white"
                )

                fig_pay.update_layout(
                    xaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                    yaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                    yaxis_tickangle=0,
                    height=400
                )

                # Menampilkan grafik jumlah pelanggan berdasarkan status
                st.plotly_chart(fig_pay, use_container_width=True)
                       
            col_ownership, col_paymethod, = st.columns(2)
            
            with col_ownership:
                # Visualisasi untuk jumlah pelanggan berdasarkan building ownership
                fig_ownership = px.bar(
                    byownership_df.sort_values(by="Building_Ownership_Status", ascending=False),
                    x="Building_Ownership_Status",
                    y="customer_count",
                    title="<b>👤Number of Customer by Building Ownership Status</b>",
                    color_discrete_sequence=["#90CAF9"] * len(byownership_df),
                    labels={"customer_count": "Number of Customers", "Building_Ownership_Status": "Pemilik"},
                    template="plotly_white"
                )

                fig_ownership.update_layout(
                    xaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                    yaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                    yaxis_tickangle=0,
                    height=400
                )

                # Menampilkan grafik jumlah pelanggan berdasarkan status
                st.plotly_chart(fig_ownership, use_container_width=True)
                
            with col_paymethod:
                fig_paymethod = px.bar(
                    bypayment_method_df.sort_values(by="Payment_Method", ascending=False),
                    x="Payment_Method",
                    y="customer_count",
                    title="<b>👤 Number of Customer by Payment Method</b>",
                    color_discrete_sequence=["#90CAF9"] * len(bypayment_method_df),
                    labels={"customer_count": "Number of Customers", "Payment_Method": "Metode Pembayaran"},
                    template="plotly_white"
                )

                fig_paymethod.update_layout(
                    xaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                    yaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                    yaxis_tickangle=0,
                    height=400
                )

                # Menampilkan grafik jumlah pelanggan berdasarkan status
                st.plotly_chart(fig_paymethod, use_container_width=True)
        

            st.subheader("Best Customer Based on RFM Parameters")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("📅 Average Recency(days)")
                avg_recency = round(rfm_df.recency.mean(), 1)
                st.metric("Average Recency (days)", value=avg_recency)
            
            with col2:
                st.info("🛒 Average Frequency")
                avg_frequency = round(rfm_df.frequency.mean(), 2)
                st.metric("Average Frequency", value=avg_frequency)
            
            with col3:
                st.info("💰 Average Monetary")
                avg_monetary = format_currency(rfm_df.monetary.mean(), "IDR", locale='id_ID') 
                st.metric("Average Monetary", value=avg_monetary)
            
            # Grafik berdasarkan Recency
            fig_recency = px.bar(
                rfm_df.sort_values(by="recency", ascending=True).head(5),
                x="Customer_Number",
                y="recency",
                title="<b>By Recency (days)</b>",
                color_discrete_sequence=["#90CAF9"],
                template="plotly_white"
            )
            fig_recency.update_layout(
                yaxis_title="Recency (days)",
                xaxis_title="Pelanggan",
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                height=400
            )

            # Grafik berdasarkan Frequency
            fig_frequency = px.bar(
                rfm_df.sort_values(by="frequency", ascending=False).head(5),
                x="Customer_Number",
                y="frequency",
                title="<b>By Frequency</b>",
                color_discrete_sequence=["#90CAF9"],
                template="plotly_white"
            )
            fig_frequency.update_layout(
                yaxis_title="Frequency",
                xaxis_title="Pelanggan",
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                height=400
            )

            # Grafik berdasarkan Monetary
            fig_monetary = px.bar(
                rfm_df.sort_values(by="monetary", ascending=False).head(5),
                x="Customer_Number",
                y="monetary",
                title="<b>By Monetary</b>",
                color_discrete_sequence=["#90CAF9"],
                template="plotly_white"
            )
            fig_monetary.update_layout(
                yaxis_title="Monetary",
                xaxis_title="Pelanggan",
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                height=400
            )

            # Menampilkan grafik di kolom Streamlit
            left, center, right = st.columns(3)
            left.plotly_chart(fig_recency, use_container_width=True)
            center.plotly_chart(fig_frequency, use_container_width=True)
            right.plotly_chart(fig_monetary, use_container_width=True)

            # Keterangan
            ket_r, ket_f, ket_m = st.columns(3, gap='small')
            with ket_r:
                st.markdown("- **Recency (R)**: Mengukur berapa lama pelanggan terakhir bertransaksi. Semakin baru transaksi, semakin tinggi nilai R.")
            with ket_f:
                st.markdown("- **Frequency (F)**: Menghitung seberapa sering pelanggan melakukan transaksi dalam periode tertentu. Semakin sering pelanggan bertransaksi, semakin tinggi nilai F.")
            with ket_m:
                st.markdown("- **Monetary (M)**: Mengukur total uang yang dihabiskan oleh pelanggan dalam periode tertentu. Semakin tinggi total pengeluaran, semakin tinggi nilai M.")
        except Exception as e:
            # st.error("Terjadi kesalahan saat membaca file atau dataset tidak sesuai")
                # Menampilkan detail traceback untuk melihat dimana error terjadi
            # Menampilkan pesan error
            print(f"Pesan error: {str(e)}")
    else:
        st.info("Silakan unggah file dataset untuk menampilkan visualisasi.")

with tab2: 
    with st.expander("Upload Dataset Income", expanded=True, icon="📤"):
        dataset2 = st.file_uploader(label = 'Upload your dataset income: ', type=['xlsx'], label_visibility='hidden')
    if dataset2 is not None:
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()
        try:
            with st.sidebar:
                all_data_clean = pd.read_excel(dataset2)
                datetime_columns = ["Tanggal_Aktif"]  # Sesuaikan dengan kolom tanggal yang ada
                all_data_clean.sort_values(by="Tanggal_Aktif", inplace=True)
                all_data_clean.reset_index(inplace=True)
                
                for column in datetime_columns:
                    all_data_clean[column] = pd.to_datetime(all_data_clean[column])
                    
                min_date_clean = all_data_clean["Tanggal_Aktif"].min()
                max_date_clean = all_data_clean["Tanggal_Aktif"].max()
                areas = all_data_clean['Area'].unique()  # Mengambil daftar area unik

                start_date_clean, end_date_clean = st.date_input(
                    label='📅 Rentang Waktu Data Pelanggan VNET',
                    min_value=min_date_clean,
                    max_value=max_date_clean,
                    value=[min_date_clean, max_date_clean]
                )
            
                selected_area = st.selectbox('🌍Pilih Area', options=areas, index=0) # arrreaa

            
            main_data_clean = all_data_clean[(all_data_clean["Tanggal_Aktif"] >= str(start_date_clean)) & 
                                            # (all_data_clean["Tanggal_Aktif"] <= str(end_date_clean + pd.Timedelta(days=1))) &
                                            (all_data_clean['Area'] == selected_area)
                                        ]

            daily_incomes_df = create_daily_incomes_df(main_data_clean)
            sum_order_items2_df = create_sum_order_items2_df(main_data_clean)
            byownerdata_df = create_byownerdata_df(main_data_clean)
            scatter_df, correlation_value = create_scatter_income_vs_duration(main_data_clean)
            customers_by_duration_area, avg_customers_per_area = create_customers_by_duration_area(main_data_clean)
            daily_incomes_ppn_df = create_daily_incomes_ppn_df(main_data_clean)
            daily_fee_seller_df = create_daily_fee_seller_df(main_data_clean)
            rfm2_df = create_rfm2_df(main_data_clean)
        
            st.subheader('Daily Installations')
            
            total1, total2, total3, = st.columns(3 ,gap='small')
            with total1:
                st.info('Total Fee Seller',icon="💰")                        
                total_fee_seller = format_currency(daily_fee_seller_df.pengeluaran.sum(), "IDR", locale='id_ID') 
                st.metric(label="Total Fee Seller", value=total_fee_seller) 
            with total2:
                st.info('Total Revenue',icon="💰")                       
                total_revenue = format_currency(daily_incomes_df.revenue.sum(), "IDR", locale='id_ID') 
                st.metric(label="Total Revenue", value=total_revenue)
            with total3:
                st.info('Total Revenue(+PPN)',icon="💰")                       
                total_revenue_ppn = format_currency(daily_incomes_ppn_df.revenue.sum(), "IDR", locale='id_ID') 
                st.metric(label="Total Revenue(+PPN)", value=total_revenue_ppn) 
                style_metric_cards(background_color="#FFFFFF",border_left_color="#29abe2",border_color="#000000",box_shadow="#F71938")    

            vis1, vis2, vis3 = st.columns(3, gap='small')            
            with vis1:
                #Membuat grafik menggunakan Plotly
                total_instals = daily_incomes_df.income_count.sum()

                fig_total = go.Figure()

                # Menambahkan total pemasangan
                fig_total.add_trace(go.Scatter(
                    x=daily_incomes_ppn_df["Tanggal_Aktif"],
                    y=daily_incomes_ppn_df["income_count"],
                    mode='lines+markers',
                    name='Total Pemasangan',
                    marker=dict(color='#90CAF9'),
                    hovertemplate="<b>Total Pemasangan: %{y}<br>Pendapatan: Rp %{customdata:,.0f}<extra></extra></b>",
                    customdata=daily_incomes_ppn_df['revenue']
                ))

                # Mengatur layout
                fig_total.update_layout(
                    title=f"<b>📈 Total Pemasangan Harian</b>",
                    xaxis_title='Tanggal',
                    yaxis_title='Total Pemasangan',
                    template="plotly_white",
                    xaxis_tickangle=-45,
                    yaxis=dict(title='Total Pemasangan', side='left', showgrid=True),
                    hovermode='x unified',
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)'
                )

                # Menampilkan grafik total pemasangan
                st.plotly_chart(fig_total, use_container_width=True)   
                st.markdown(f"- **Total Pemasangan sebanyak {total_instals}**")
            with vis2:
                # Menggunakan Plotly Express untuk membuat bar chart
                fig = px.bar(
                    customers_by_duration_area,
                    x="Lama_Berlangganan(Bulan)",
                    y="customer_count",
                    color="Area",
                    title="👤 Jumlah Pelanggan Berdasarkan Lama Berlangganan per Area",
                    labels={"Lama_Berlangganan(Bulan)": "Lama Berlangganan (Bulan)", "customer_count": "Jumlah Pelanggan Yang aktivasi"},
                    barmode='stack',  # Menyusun bar secara bertumpuk
                )

                # Menambahkan grid
                fig.update_layout(
                    xaxis_title="Lama Berlangganan (Bulan)",
                    yaxis_title="Jumlah Pelanggan Yang Aktivasi",
                    legend_title="Area",
                    template="plotly_white",  # Menggunakan template putih
                )
                
                # Tampilkan plot di Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
                # Menghitung rata-rata lama berlangganan
                average_duration = customers_by_duration_area['Lama_Berlangganan(Bulan)'].mean()

                # Menampilkan hasil rata-rata di Streamlit
                st.markdown(f"- **Rata-rata Lama Berlangganan Pelanggan: {average_duration:.0f} bulan**")
            
            with vis3: 
                fig = px.scatter(
                scatter_df,
                x='Lama_Berlangganan(Bulan)',
                y='Pendapatan_Bersih',
                title='👤 Lama Berlangganan vs Pendapatan Bersih',
                labels={
                    'Lama_Berlangganan(Bulan)': 'Lama Berlangganan (Bulan)',
                    'Pendapatan_Bersih': 'Pendapatan Bersih'
                },
                color_discrete_sequence=['#90CAF9']
                )

                # Tampilkan plot
                st.plotly_chart(fig)

                # Menentukan keterangan berdasarkan nilai korelasi
                if correlation_value > 0:
                    keterangan = f"Nilai korelasi antara Lama Berlangganan (Bulan) dan Pendapatan Bersih adalah: {correlation_value:.2f}. Korelasi Positif : Menunjukkan bahwa semakin lama pelanggan berlangganan, semakin besar pendapatan yang dihasilkan."
                elif correlation_value < 0:
                    keterangan = f"Nilai korelasi antara Lama Berlangganan (Bulan) dan Pendapatan Bersih adalah: {correlation_value:.2f}. Korelasi Negatif : Menunjukkan bahwa semakin lama pelanggan berlangganan, semakin kecil pendapatan yang dihasilkan."
                elif correlation_value == 0:
                    keterangan = f"Nilai korelasi antara Lama Berlangganan (Bulan) dan Pendapatan Bersih adalah: {correlation_value:.2f}. Korelasi 0: Menunjukan tidak ada hubungan linear antara lama berlangganan dan pendapatan."
                else:
                    keterangan = "Nilai korelasi tidak ada"
                # Menampilkan keterangan korelasi
                st.markdown(f"- **{keterangan}**")
            
            totalrev1, totalrev2, totalrev3, = st.columns(3 ,gap='small')
            with totalrev1:
                st.info('Income Terkecil',icon="💰")
                min_revenue_ppn = format_currency(daily_incomes_ppn_df.revenue.min(), "IDR", locale='id_ID') 
                st.metric("Income Terkecil", value=min_revenue_ppn)
                
            with totalrev2:
                st.info('Income Rata-Rata',icon="💰")
                avg_revenue_ppn = format_currency(daily_incomes_ppn_df.revenue.mean(), "IDR", locale='id_ID') 
                st.metric("Income Rata-Rata", value=avg_revenue_ppn)   
            
            with totalrev3:
                st.info('Income Tertinggi',icon="💰")
                max_revenue_ppn = format_currency(daily_incomes_ppn_df.revenue.max(), "IDR", locale='id_ID') 
                st.metric("Income Tertinggi", value=max_revenue_ppn)   

            # Menampilkan grafik Best & Worst Performing Product
            col1, col2, col3, col4 = st.columns(4, gap='small')

            with col1:
                best_product = sum_order_items2_df.head(5)
                fig_best = px.bar(
                    best_product,
                    x="quantity_x",
                    y="Paket_Langganan",
                    orientation='h',
                    title="<b>📈 Best Performing Product</b>",
                    color_discrete_sequence=["#90CAF9"] * len(best_product),
                    labels={"quantity_x": "Jumlah Pelanggan Yang Aktivasi", "Paket_Langganan": "Paket Langganan"},
                    template="plotly_white"
                )
                fig_best.update_layout(
                    xaxis=dict(title="Number of Sales", showgrid=True, gridcolor='#cecdcd'),
                    yaxis=dict(title=None, showgrid=False),
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                )

                # Menampilkan grafik best performing
                st.plotly_chart(fig_best, use_container_width=True)
            with col2:
                # Grafik untuk Worst Performing Product
                worst_product = sum_order_items2_df.sort_values(by="quantity_x", ascending=True).head(5)
                fig_worst = px.bar(
                    worst_product,
                    x="quantity_x",
                    y="Paket_Langganan",
                    orientation='h',
                    title="<b>📉 Worst Performing Product</b>",
                    color_discrete_sequence=["#90CAF9"] * len(worst_product),
                    labels={"quantity_x": "Jumlah Pelanggan Yang Aktivasi", "Paket_Langganan": "Paket Langganan"},
                    template="plotly_white"
                )
                fig_worst.update_layout(
                    xaxis=dict(title="Number of Sales", showgrid=True, gridcolor='#cecdcd'),
                    yaxis=dict(title=None, showgrid=False),
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                )

                # Menampilkan grafik worst performing
                st.plotly_chart(fig_worst, use_container_width=True)

            with col3:
                # Grafik untuk 5 teratas
                top_customers = byownerdata_df.sort_values(by="customer_count", ascending=False).head(5)
                fig_top = px.bar(
                    top_customers,
                    x="customer_count",
                    y="Owner_Data",
                    orientation='h',
                    title="<b>📈 TOP 5 CUSTOMERS BY OWNER DATA</b>",
                    color_discrete_sequence=["#90CAF9"] * len(top_customers),
                    labels={"customer_count": "Jumlah Pelanggan Yang Aktivasi", "Owner_Data": "Owner Data"},
                    template="plotly_white"
                )
                fig_top.update_layout(
                    yaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                    xaxis_title="Customer Count",
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                )

                # Menampilkan grafik top customers
                st.plotly_chart(fig_top, use_container_width=True)
            with col4:
                # Grafik untuk 5 terbawah
                bottom_customers = byownerdata_df.sort_values(by="customer_count", ascending=True).head(5)
                fig_bottom = px.bar(
                    bottom_customers,
                    x="customer_count",
                    y="Owner_Data",
                    orientation='h',
                    title="<b>📉 BOTTOM 5 CUSTOMERS BY OWNER DATA</b>",
                    color_discrete_sequence=["#90CAF9"] * len(bottom_customers),
                    labels={"customer_count": "Jumlah Pelanggan Yang Aktivasi", "Owner_Data": "Owner Data"},
                    template="plotly_white"
                )
                fig_bottom.update_layout(
                    yaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                    xaxis_title="Customer Count",
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                )

                # Menampilkan grafik bottom customers
                st.plotly_chart(fig_bottom, use_container_width=True)

            # Grafik untuk Customer Count berdasarkan Owner Data
            fig_customer = px.bar(
                byownerdata_df.sort_values(by="customer_count", ascending=False),
                x="customer_count",
                y="Owner_Data",
                orientation='h',
                title="<b>NUMBER OF CUSTOMER BY OWNER DATA</b>",
                color_discrete_sequence=["#90CAF9"] + ["#D3D3D3"] * (len(byownerdata_df) - 1),
                template="plotly_white"
            )
            fig_customer.update_layout(
                yaxis=dict(title=None, showgrid=True, gridcolor='#cecdcd'),
                xaxis_title="Customer Count",
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
            )

            st.subheader("Best Customer Based on RFM Parameters")
            avgrecency, avgfrequency, avgmonetary= st.columns(3)

            with avgrecency:
                st.info("📅 Average Recency(days)")
                avg_recency = round(rfm2_df.recency.mean(), 1)
                st.metric("Rata-rata Recency (hari)", value=avg_recency)

            with avgfrequency:
                st.info("🛒 Average Frequency")
                avg_frequency = round(rfm2_df.frequency.mean(), 2)
                st.metric("Rata-rata Frequency", value=avg_frequency)

            with avgmonetary:
                st.info("💰 Average Monetary")
                avg_monetary = format_currency(rfm2_df.monetary.mean(), "IDR", locale='id_ID')
                st.metric("Rata-rata Monetary", value=avg_monetary)

            # Grafik berdasarkan Recency
            fig_recency = px.bar(
                rfm2_df.sort_values(by="recency", ascending=True).head(5),
                x="ID_Pelanggan",
                y="recency",
                title="<b>By Recency (days)</b>",
                color_discrete_sequence=["#90CAF9"],
                template="plotly_white"
            )
            fig_recency.update_layout(
                yaxis_title="Recency (days)",
                xaxis_title="ID_Pelanggan",
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
            )

            # Grafik berdasarkan Frequency
            fig_frequency = px.bar(
                rfm2_df.sort_values(by="frequency", ascending=False).head(5),
                x="ID_Pelanggan",
                y="frequency",
                title="<b>By Frequency</b>",
                color_discrete_sequence=["#90CAF9"],
                template="plotly_white"
            )
            fig_frequency.update_layout(
                yaxis_title="Frequency",
                xaxis_title="ID_Pelanggan",
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
            )

            # Grafik berdasarkan Monetary
            fig_monetary = px.bar(
                rfm2_df.sort_values(by="monetary", ascending=False).head(5),
                x="ID_Pelanggan",
                y="monetary",
                title="<b>By Monetary</b>",
                color_discrete_sequence=["#90CAF9"],
                template="plotly_white"
            )
            fig_monetary.update_layout(
                yaxis_title="Monetary",
                xaxis_title="ID_Pelanggan",
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
            )

            left,right,center=st.columns(3)
            left.plotly_chart(fig_recency,use_container_width=True)
            right.plotly_chart(fig_frequency,use_container_width=True)
            center.plotly_chart(fig_monetary,use_container_width=True)
            
            ket_r, ket_f, ket_m, = st.columns(3,gap='small')
            with ket_r:
                st.markdown("- **Recency (R)**: Mengukur berapa lama pelanggan terakhir bertransaksi. Semakin baru transaksi, semakin tinggi nilai R.")
            with ket_f:
                st.markdown("- **Frequency (F)**: Menghitung seberapa sering pelanggan melakukan transaksi dalam periode tertentu. Semakin sering pelanggan bertransaksi, semakin tinggi nilai F.")
            with ket_m:
                st.markdown("- **Monetary (M)**: Mengukur total uang yang dihabiskan oleh pelanggan dalam periode tertentu. Semakin tinggi total pengeluaran, semakin tinggi nilai M.")
            
            st.header("Segmentasi Pelanggan")
            
            with st.expander("ANALISIS RFM, DETEKSI OUTLIER, CAPPING, DAN PCA"):
                
                col_rfm, col_out1, col_out2, = st.columns(3)
                with col_rfm:
                    st.subheader("RFM Data Pelanggan")
                    st.dataframe(rfm2_df)  # Menampilkan RFM data asli sebelum capping
                with col_out1:
                    # OUTLIER SEBELUM CAPPING
                    st.subheader("Outlier Pada Data Sebelum Capping")
                    # check_outliers_rfm_streamlit(rfm2_df)  
                    check_outliers_rfm_streamlit(rfm2_df)  
                # CAPPIING
                    rfm2_df_capped = cap_outliers_rfm_streamlit(rfm2_df) 

                # Cek kembali outlier setelah dilakukan capping
                    st.subheader("Outlier Pada Data Setelah Capping")
                    # check_outliers_rfm_streamlit(rfm2_df_capped)
                    check_outliers_rfm_streamlit(rfm2_df_capped)
                with col_out2:
                    #Membuat pipeline untuk scaling dan PCA
                    preprocessor = Pipeline(
                        [
                            ("scaler", MinMaxScaler()),  # Scaling dengan MinMaxScaler
                            ("pca", PCA(n_components=2, random_state=42)),  # PCA untuk mereduksi ke 2 dimensi
                        ]
                    )
                    
                    X = rfm2_df_capped.drop('ID_Pelanggan',axis=1)
                    X_scaled = pd.DataFrame(preprocessor.fit_transform(X),columns=['PC_1','PC_2'])
                    
                    st.subheader("Hasil Dimensionality Reduction (PCA)")
                    st.dataframe(X_scaled)
                    
            with st.spinner('Wait for it...'):
                time.sleep(3)
                
            with st.expander("Statistik Hopkins dan Pencarian Nilai K yang optimal"):
                col_hop_elbow, col_sil, = st.columns(2)        
                # Menghitung Hopkins Statistic untuk X_scaled
                with col_hop_elbow:
                    st.subheader("Statistik Hopkins")
                    st.markdown("""
                                Statistik Hopkins digunakan untuk mengecek kecenderungan klaster, dengan kata lain: seberapa baik data dapat dikelompokkan.
                                - **Jika nilainya antara {0.01, ...,0.3}**: data memiliki jarak yang teratur.
                                - **Jika nilainya sekitar 0,5**: data tersebut acak.
                                - **Jika nilainya antara {0.7, ..., 0.99}**: data tersebut memiliki kecenderungan tinggi untuk mengelompok.
                                """)
                    hopkins_values = [round(hopkins(X_scaled[['PC_1', 'PC_2']]), 3) for _ in range(5)]

                    # Menampilkan hasil Hopkins Statistic pada Streamlit
                    st.subheader("Hopkins Statistic Test")
                    for i, val in enumerate(hopkins_values, 1):
                        st.write(f"Percobaan {i}: Hopkins statistic value is {val}")
                        
                with col_sil:
                    st.subheader("Pencarian nilai (K) yang optimal")
                    # elbow-curve/SSD
                    ssd = []
                    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
                    for num_clusters in range_n_clusters:
                        kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
                        kmeans.fit(X_scaled)
                        ssd.append(kmeans.inertia_)

                    plt.figure(figsize=(20, 12))
                    plt.plot(range_n_clusters, ssd, marker='o')
                    plt.xlabel('Number of clusters (k)')
                    plt.ylabel('Sum of Squared Distances (SSD)')
                    plt.title('Elbow Curve')
                    plt.xticks(range_n_clusters)
                    plt.grid()
                    st.pyplot(plt)  # Menampilkan plot di Streamlit
                    # silhouette analysis
                    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

                    for num_clusters in range_n_clusters:
                        
                        # intialise kmeans
                        kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
                        kmeans.fit(X_scaled)
                        
                        cluster_labels = kmeans.labels_
                        
                        # silhouette score
                        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                        st.write("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, round(silhouette_avg,2)))
            
            with st.spinner('Wait for it...'):
                time.sleep(2)
                        
            with st.expander("Clustering K-Means dan Persebaran Data"):
                clus1, clus2, = st.columns(2)
                with clus1:
                    st.header("Clustering K-Means dengan K=3")
                # Run K-means using optimal K
                    # final model with k=3
                    kmeans = KMeans(n_clusters=3, max_iter=50,random_state=1)
                    kmeans.fit(X_scaled)

                    # Adding cluster labels to master dataframe
                    X_scaled['cluster_id'] = kmeans.labels_
                    X['cluster_id'] = kmeans.labels_
                    
                    cluster_counts = X_scaled['cluster_id'].value_counts().reset_index()
                    cluster_counts.columns = ['cluster_id', 'number_of_customers']  
                    # Menyusun nama cluster
                    cluster_labels = {0: 'Loyal', 1: 'Promising', 2: 'Need Attention'}
                    cluster_counts['cluster_label'] = cluster_counts['cluster_id'].map(cluster_labels)
                    
                    # Membuat bar chart menggunakan plotly.express
                    fig = px.bar(cluster_counts, x='cluster_label', y='number_of_customers',
                                title='Customer Distribution',
                                labels={'number_of_customers': 'Number of Customers', 'cluster_label': 'Clusters'},
                                color='cluster_label',
                                color_discrete_sequence=px.colors.qualitative.Plotly)

                    # Menampilkan plot di Streamlit
                    st.plotly_chart(fig)


                with clus2:
                    st.header("Clustering Profiling Using R-F-M")
                    # Visualizing Numerical columns using Boxplots
                    cols = X.columns[0:-1].tolist()

                    # Membuat figure untuk boxplots
                    fig_box = sp.make_subplots(rows=1, cols=len(cols), subplot_titles=cols)

                    for n, col in enumerate(cols):
                        box_fig = px.box(X, x='cluster_id', y=col, title=f'Boxplot of {col}')
                        for trace in box_fig.data:
                            fig_box.add_trace(trace, row=1, col=n + 1)

                    fig_box.update_layout(height=400, title_text='Boxplots for Clustering Variables')
                    st.plotly_chart(fig_box)
                
                fig = plt.figure(figsize=[20,8])

                clus3, clus4, = st.columns(2)
                
                with clus3:
                    plt.subplot(1,3,1)
                    sns.scatterplot(data=X,x="recency",y="frequency",hue="cluster_id",size="cluster_id",palette="Set1")
                    plt.subplot(1,3,2)
                    sns.scatterplot(data=X,x="frequency",y="monetary",hue="cluster_id",size="cluster_id",palette="Set1")
                    plt.subplot(1,3,3)
                    sns.scatterplot(data=X,x="monetary",y="recency",hue="cluster_id",size="cluster_id",palette="Set1")
                    st.pyplot(plt)

                with clus4:
                    # Vusializing clusters using Principle Components
                    fig = plt.figure(figsize=[20,8])

                    sns.scatterplot(data=X_scaled,x="PC_1",y="PC_2",hue="cluster_id",size="cluster_id",palette="Set1")
                    st.pyplot(plt)

            with st.spinner('Wait for it...'):
                time.sleep(2)
            
            with st.expander("Hasil Segementasi"):
                # Gabungkan data X_scaled dengan ID pelanggan dari RFM
                rfm_with_cluster = pd.concat([rfm2_df_capped, X_scaled[['PC_1', 'PC_2', 'cluster_id']]], axis=1)
                
                jum_clas, vis_col, = st.columns(2)
                with jum_clas:
                    st.subheader("Jumlah Pelanggan per Cluster")
                    st.dataframe(cluster_counts)   

                    # Menampilkan pie chart untuk melihat distribusi segmen
                    # Hitung jumlah pelanggan di setiap cluster
                    rfm_with_cluster['segment'] = rfm_with_cluster['cluster_id'].map({
                        0: 'Loyal',
                        1: 'Promising',
                        2: 'Need Attention'
                    })

                    # Menghitung jumlah pelanggan berdasarkan segmen
                    cluster_counts = rfm_with_cluster['segment'].value_counts().reset_index()
                    cluster_counts.columns = ['Segment', 'Jumlah Pelanggan']

                    # Membuat pie chart menggunakan Plotly Express
                    fig = px.pie(
                        cluster_counts, 
                        values='Jumlah Pelanggan', 
                        names='Segment', 
                        title='Distribusi Pelanggan Berdasarkan Segmen',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_traces(textinfo='percent+label')

                    # Menampilkan pie chart di Streamlit
                    st.plotly_chart(fig)    
                
                with vis_col:                
                    # Filter berdasarkan cluster
                    selected_cluster = st.selectbox('Pilih Cluster untuk Ditampilkan:', rfm_with_cluster['cluster_id'].unique())

                    # Filter data berdasarkan cluster yang dipilih
                    filtered_data = rfm_with_cluster[rfm_with_cluster['cluster_id'] == selected_cluster]

                    # Tampilkan hasil filter di Streamlit
                    st.subheader(f"Data Pelanggan di Cluster {selected_cluster}")
                    st.dataframe(filtered_data)
                    
                    # Hitung total jumlah pelanggan di cluster yang dipilih
                    total_pelanggan_cluster = filtered_data.shape[0]
                    
                    # Tampilkan total jumlah pelanggan di Streamlit
                    st.write(f"Total Pelanggan di Cluster {selected_cluster}: {total_pelanggan_cluster}")
                    
                    excel_data = convert_df_to_excel(rfm_with_cluster)

                    # Menambahkan tombol untuk mengunduh data dalam format .xlsx
                    st.download_button(
                        label="⭐ Unduh Semua Data Pelanggan Segmented",
                        data=excel_data,
                        file_name='pelanggan_segmented.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                       
        except Exception as e:
            # st.error("Terjadi kesalahan saat membaca file atau dataset tidak sesuai")
            st.error(e)
    else:
        st.info("Silakan unggah file dataset untuk menampilkan visualisasi.")

#----- HIDE STREAMLIT STYLE -----
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """    
st.markdown(hide_st_style, unsafe_allow_html=True)
