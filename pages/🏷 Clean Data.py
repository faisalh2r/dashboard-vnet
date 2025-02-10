import streamlit as st
import pandas as pd
from io import BytesIO 
import time

st.set_page_config(page_title = 'File Uploader') 

# load Style css
with open('css/style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

st.sidebar.image("images/logo.png",caption="")

def clean_data_pelanggan(df):
    try:
        pelanggan_df = df.copy()
        
        st.write("Data sebelum pembersihan:")
        st.write(pelanggan_df.head())  #
        # WRANGLING
        pelanggan_df.columns = pelanggan_df.columns.str.replace(' ', '_')

        if 'Registration_Number' not in pelanggan_df.columns:
            st.error("Terjadi kesalahan saat membaca file, Silahkan upload dataset yang sesuai")
            st.stop()  # Menghentikan eksekusi jika kolom tidak ada
                
        st.write(pelanggan_df.isna().sum())

        # Cek data NaN
        nan_results = pelanggan_df.isna().sum()
        if nan_results.any():
            st.warning("Terdapat nilai NaN di kolom berikut:")
            st.write(nan_results[nan_results > 0])
            
        # Cek data duplikat
        dup_results = pelanggan_df.duplicated().sum()
        if dup_results > 0:
            st.warning("Terdapat nilai Duplikat Sebanyak:")
            st.write(dup_results)
            pelanggan_df = pelanggan_df.drop_duplicates()
            st.success(f"{dup_results} baris duplikat dihapus.")

        
        # Instalattion date, Start Period, End Period, Due Date, Confirm Date
        datetime_columns = ["Installation_Date", "Start_Period", "End_Period", "Due_Date", "Confirm_Date"]
        
        for column in datetime_columns:
            pelanggan_df[column] = pd.to_datetime(pelanggan_df[column])

        st.write("Nama kolom dalam DataFrame:")
        st.write(pelanggan_df.columns)                
        
        # Mendefinisikan tipe data yang diharapkan
        expected_types = {
            'Billing_Number': 'object',
            'Type': 'object',        
            'Name': 'object',               
            'Registration_Number': 'object',
            'Customer_Number': 'object',
            'Building_Ownership_Status': 'object',
            'Installation_Date': 'datetime64[ns]',
            'Project': 'object',
            'Status': 'object',
            'Payment_Method': 'object',
            'Start_Period': 'datetime64[ns]',
            'End_Period': 'datetime64[ns]',
            'Due_Date': 'datetime64[ns]',
            'Confirm_Date': 'datetime64[ns]',
            'Internet_Package': 'object',
            'Installation_Cost': 'Int64',  
            'Additional_Amount': 'Int64',  
            'Admin_Fee': 'Int64', 
            'Deposit_Amount': 'Int64', 
            'Additional_Item': 'float64',
            'Final_Amount': 'Int64'
        }    
        
        # Memastikan tipe data sesuai dengan yang diharapkan
        for column, expected_type in expected_types.items():
            actual_type = str(pelanggan_df[column].dtype)
            # Cek dan ubah tipe data jika tidak sesuai
            if expected_type != actual_type:
                try:
                    if expected_type == 'datetime64[ns]':
                        pelanggan_df[column] = pd.to_datetime(pelanggan_df[column], errors='coerce')
                    elif expected_type == 'Int64':  # Untuk nullable integer
                        pelanggan_df[column] = pd.to_numeric(pelanggan_df[column], errors='coerce').astype('Int64')
                    else:
                        pelanggan_df[column] = pelanggan_df[column].astype(expected_type)
                except Exception as e:
                    st.warning(f"Gagal mengubah kolom '{column}' menjadi '{expected_type}': {e}")
                    
        # Menampilkan tipe data setiap kolom di dataframe
        st.write("Tipe data untuk setiap kolom:")
        st.write(pelanggan_df.dtypes)
        
        pelanggan_df['Name'].fillna("Tidak Diketahui", inplace=True)
            
        def fill_installation_date(row):
            if pd.notna(row['Start_Period']):
                return row['Start_Period']
            elif pd.notna(row['Due_Date']):
                return row['Due_Date']
            else:
                return pd.NaT  # Kembali ke NaT jika keduanya kosong

        # Terapkan fungsi ke setiap baris
        pelanggan_df['Installation_Date'] = pelanggan_df.apply(fill_installation_date, axis=1)
        # Menghapus baris dengan NaN di kolom penting
        
        pelanggan_df['Start_Period'] = pelanggan_df['Start_Period'].fillna(pelanggan_df['Due_Date'] - pd.Timedelta(days=1))

        # Isi 'End Period' dengan 'Start Period' + 1 bulan
        pelanggan_df['End_Period'] = pelanggan_df['Start_Period'] + pd.DateOffset(months=1)
        
        pelanggan_df['Payment_Method'] = pelanggan_df['Payment_Method'].fillna("None")
        pelanggan_df['Customer_Number'] = pelanggan_df['Customer_Number'].fillna("None")
        
        pelanggan_df['Owner'] = pelanggan_df["Internet_Package"].apply(lambda x: "Viberlink" if x in ['LITE link 100Mbps - 30d', 'Promo Free 100Mbps '] else "VNET")
        
                # Fungsi untuk menentukan status
        def determine_status(row):
            if row['Type'] == 'CUSTOMER':
                return 'Active'
            else:
                return 'Non-Active'

        # Terapkan fungsi ke setiap baris
        pelanggan_df['Status_Customer'] = pelanggan_df.apply(determine_status, axis=1)
    
        
        st.write("Data setelah pembersihan:")
        st.write(pelanggan_df.head())

        st.write(pelanggan_df.isna().sum())
        
        # pelanggan_df.dropna(subset=['Additional_Item'], inplace=True)

        # bikin kolom baru owner untuk memisahkan pelanggan 
        
        st.write("Cek setelah fillna:")
        st.write(pelanggan_df.head())
        
    except Exception as e:
        st.error(str(e))   
    return pelanggan_df

# def clean_data_pelanggan(df):
#     try:
#         pelanggan_df = df.copy()
#         # WRANGLING
#         pelanggan_df.columns = pelanggan_df.columns.str.replace(' ', '_')
        
#         if 'No._Reg' not in pelanggan_df.columns:
#             st.error("Terjadi kesalahan saat membaca file, Silahkan upload dataset yang sesuai")
#             st.stop()  # Menghentikan eksekusi jika kolom tidak ada
        
#         pelanggan_df = pelanggan_df.rename(columns={'No._Reg': 'No_Reg'})
        
#         # Cek data NaN
#         nan_results = pelanggan_df.isna().sum()
#         if nan_results.any():
#             st.warning("Terdapat nilai NaN di kolom berikut:")
#             st.write(nan_results[nan_results > 0])
#             pelanggan_df['Nama'].fillna("Tidak Diketahui", inplace=True)
#             # Menghapus baris dengan NaN di kolom penting
#             pelanggan_df.dropna(subset=['Biaya', 'Durasi', 'Tanggal_Terpasang', 'Tanggal_Jatuh_Tempo'], inplace=True)
#             # Mengganti NaN di kolom 'Durasi' dengan median
#             pelanggan_df['Durasi'].fillna(pelanggan_df['Durasi'].median(), inplace=True)
            
#         # Cek data duplikat
#         dup_results = pelanggan_df.duplicated().sum()
#         if dup_results > 0:
#             st.warning("Terdapat nilai Duplikat Sebanyak:")
#             st.write(dup_results)
#             pelanggan_df = pelanggan_df.drop_duplicates()
#             st.success(f"{dup_results} baris duplikat dihapus.")
            
#         # Ubah tipe data Biaya dan Deposit
#         biaya_deposit_columns = ['Biaya', 'Deposit']

#         for col in biaya_deposit_columns:
#             # Pastikan kolom adalah tipe string
#             pelanggan_df[col] = pelanggan_df[col].astype(str)
#             # Menghilangkan 'Rp', koma, titik, dan spasi tambahan jika ada
#             pelanggan_df[col] = pelanggan_df[col].str.replace('Rp', '', regex=False)
#             pelanggan_df[col] = pelanggan_df[col].str.replace(',', '', regex=False)
#             pelanggan_df[col] = pelanggan_df[col].str.replace('.', '', regex=False)
#             pelanggan_df[col] = pelanggan_df[col].str.strip()  # Menghapus spasi di depan/belakang
#             # Mengonversi tipe data menjadi numeric
#             pelanggan_df[col] = pd.to_numeric(pelanggan_df[col], errors='coerce')
#             # Mengganti NaN yang dihasilkan dari konversi dengan 0
#             pelanggan_df[col].fillna(0, inplace=True)
#             pelanggan_df[col] = pelanggan_df[col].astype('Int64')  # Menggunakan Int64 untuk nullable

#         # Mendefinisikan tipe data yang diharapkan
#         expected_types = {
#             'No_Reg': 'object',
#             'Tanggal_Terpasang': 'datetime64[ns]',
#             'Tanggal_Jatuh_Tempo': 'datetime64[ns]',
#             'Nama': 'object',
#             'Paket': 'object',
#             'Durasi': 'int64',
#             'Biaya': 'Int64',  # Menggunakan Int64 untuk nullable
#             'Deposit': 'Int64',  # Menggunakan Int64 untuk nullable
#             'Status': 'object'
#         }    
        
#         # Memastikan tipe data sesuai dengan yang diharapkan
#         for column, expected_type in expected_types.items():
#             actual_type = str(pelanggan_df[column].dtype)
#             # Cek dan ubah tipe data jika tidak sesuai
#             if expected_type != actual_type:
#                 try:
#                     if expected_type == 'datetime64[ns]':
#                         pelanggan_df[column] = pd.to_datetime(pelanggan_df[column], errors='coerce')
#                     elif expected_type == 'Int64':  # Untuk nullable integer
#                         pelanggan_df[column] = pd.to_numeric(pelanggan_df[column], errors='coerce').astype('Int64')
#                     else:
#                         pelanggan_df[column] = pelanggan_df[column].astype(expected_type)
#                 except Exception as e:
#                     st.warning(f"Gagal mengubah kolom '{column}' menjadi '{expected_type}': {e}")

#         # Standarisasi Nama            
#         pelanggan_df['Nama'] = pelanggan_df['Nama'].str.title()
#         # EDA: Menghitung lama berlangganan dalam hari
#         pelanggan_df['Lama_Berlangganan'] = (pelanggan_df['Tanggal_Jatuh_Tempo'] - pelanggan_df['Tanggal_Terpasang']).dt.days
#         # Menandai churn (jika status adalah isolir atau dismantle)
#         pelanggan_df['Churn'] = pelanggan_df['Status'].apply(lambda x: "Ya" if x in ['isolir', 'dismantle', 'dismantle trial'] else "Tidak")
#         pelanggan_df['Churn_numeric'] = pelanggan_df['Churn'].map({'Ya': 1, 'Tidak': 0})
#     except Exception as e:
#         st.error(str(e))   
#     return pelanggan_df
    
def clean_data_incomes (df):
    try:
        # List untuk menyimpan dataset setelah 4 baris pertama dihapus
        cleaned_dfs = []

        for single_df in df:
            # Hapus baris 1-4 dan reset index
            df_cleaned = single_df.iloc[3:].reset_index(drop=True)
            
            # Set baris pertama (yang sekarang ada di index 0) sebagai header
            df_cleaned.columns = df_cleaned.iloc[0]
            
            # Hapus index
            df_cleaned = df_cleaned.drop(index=0).reset_index(drop=True)
            
            # Ganti spasi "" dengan  "_"
            df_cleaned.columns = df_cleaned.columns.str.replace(' ', '_', regex=False)

            cleaned_dfs.append(df_cleaned)
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
    return df_cleaned  
    
def extract_area(paket):
    # Deteksi area berdasarkan kolom "Paket_Langganan"
    if "Sukabumi" in paket or "SKMI" in paket or "SMI" in paket or "Sukabum" in paket:
        return "Sukabumi"
    elif "Bandung" in paket or "Paskal" in paket or "Hotspot" in paket:
        return "Bandung"
    else:
        return "Jakarta"  
    
def clean_merged_data(df):
    # Cek dan hapus duplikat setelah dataset digabung
    if 'Tanggal_Aktif' not in df.columns:
        st.error("Terjadi kesalahan saat membaca file, Silahkan upload dataset yang sesuai")
        st.stop()  # Menghentikan eksekusi jika terjadi kesalahan

    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        st.write("Jumlah baris sebelum penghapusan duplikat:", len(df))
        st.warning(f"Terdapat {num_duplicates} baris duplikat di dataset gabungan. Menghapus duplikat...")
        df = df.drop_duplicates()
        st.write("Jumlah baris setelah penghapusan duplikat:", len(df))

    # Konversi kolom 'Tanggal_Aktif' ke datetime64[ns]
    df['Tanggal_Aktif'] = pd.to_datetime(df['Tanggal_Aktif'], errors='coerce')
    # Fungsi untuk menghitung lama berlangganan (selisih bulan)
    def calculate_lama_berlangganan(dates):
        if len(dates) > 0:
            # Dapatkan bulan dari tanggal pertama dan terakhir
            first_date = dates.min()  # Tanggal pertama aktivasi
            last_date = dates.max()   # Tanggal terakhir aktivasi
            # Hitung perbedaan bulan (termasuk tahun jika berbeda)
            lama_bulan = (last_date.year - first_date.year) * 12 + last_date.month - first_date.month + 1
            return lama_bulan
        return 0

    # Kelompokkan berdasarkan ID_Pelanggan dan hitung lama berlangganan
    df['Lama_Berlangganan(Bulan)'] = df.groupby('ID_Pelanggan')['Tanggal_Aktif'].transform(calculate_lama_berlangganan)
    # Hitung Pendapatan Bersih
    df['Pendapatan_Bersih'] = df['Total'] - (
        df['Fee_Seller'] + 
        df['PPN']
    )

    # Cek nilai null dan isi kolom 'Nama' dan 'Owner_Data' yang kosong dengan 'Tidak Diketahui'
    nan_results = df.isna().sum()
    if nan_results.any():
        st.warning("Terdapat nilai NaN di kolom berikut pada dataset:")
        st.write(nan_results[nan_results > 0])

    df['Nama'].fillna("Tidak Diketahui", inplace=True)
    df['Owner_Data'].fillna("Tidak Diketahui", inplace=True)
    # Hapus baris yang masih memiliki nilai null setelah pengisian default
    df = df.dropna(how='any')
    # Tambahkan kolom "Area" berdasarkan kolom "Paket_Langganan"
    df['Area'] = df['Paket_Langganan'].apply(extract_area)

    return df

def clean_merged_data2(df):
    # Cek data duplikat
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        st.write("Jumlah baris sebelum penghapusan duplikat:", len(df))
        st.warning(f"Terdapat {num_duplicates} baris duplikat di dataset gabungan. Menghapus duplikat...")
        df = df.drop_duplicates()
        st.write("Jumlah baris setelah penghapusan duplikat:", len(df))
    return df

st.title("Upload dan Cleaning Dataset")
uploaded_files = st.file_uploader("📤 Unggah Dataset (bisa lebih dari satu)", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    # Menyimpan DataFrame dalam dictionary untuk menyimpan setiap dataset
    all_dfs = {}
        
    # Membaca semua file yang diunggah
    for uploaded_file in uploaded_files:
        # Membaca semua sheet dari file
        if uploaded_file.name.endswith('.xls'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None, engine='xlrd')
        else:
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            
        for sheet_name, df in all_sheets.items():
            all_dfs[f"{uploaded_file.name} - {sheet_name}"] = df
        
    selected_datasets = st.multiselect("Pilih Dataset untuk Dibersihkan:", list(all_dfs.keys()))
    # Pilih dataset yang akan dibersihkan
    # Memilih jenis cleaning
    cleaning_option = st.selectbox("Pilih Jenis Cleaning:", ["Cleaning Dataset VNET dan Viberlink", "Cleaning Dataset VNET(Sukabumi, Bandung, dan Jakarta)"])
    # Tombol untuk memulai pembersihan
    if st.button("🫧 Pembersihan Data"):
        cleaned_dfs = []  # Dictionary untuk menyimpan DataFrame yang sudah dibersihkan
            
        for selected_dataset in selected_datasets:
            df_to_clean = all_dfs[selected_dataset]

            if cleaning_option == "Cleaning Dataset VNET(Sukabumi, Bandung, dan Jakarta)":
                cleaned_df = clean_data_incomes([df_to_clean])
                file_name = "data_vnet_bersih.xlsx"
                cleaned_dfs.append(cleaned_df)  # Simpan DataFrame yang sudah dibersihkan
                all_data = pd.concat(cleaned_dfs, ignore_index=True)
                all_data = clean_merged_data(all_data) 

            if cleaning_option == "Cleaning Dataset VNET dan Viberlink":
                cleaned_df = clean_data_pelanggan(df_to_clean)
                file_name = "data_pelanggan_vnet_viberlink_bersih.xlsx"
                cleaned_dfs.append(cleaned_df)  # Simpan DataFrame yang sudah dibersihkan
                all_data= pd.concat(cleaned_dfs, ignore_index=True)
                all_data = clean_merged_data2(all_data)     
            
        # Simpan DataFrame gabungan yang sudah dibersihkan ke file Excel
        output_combined = BytesIO()
        all_data.to_excel(output_combined, index=False)
        output_combined.seek(0)

        # Tombol untuk mengunduh dataset gabungan yang sudah dibersihkan
        st.download_button(
            label="⭐ Unduh Semua Data yang Sudah Dibersihkan",
            data=output_combined,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
#----- HIDE STREAMLIT STYLE -----
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """    

st.markdown(hide_st_style, unsafe_allow_html=True)
