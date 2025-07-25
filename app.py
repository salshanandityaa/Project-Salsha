import os #interaksi dgn sistem
from datetime import datetime # bekerja dgn tanggal dan waktu
import pandas as pd # pustaka utk manipulasi, analisis data, dan membaca excel
import numpy as np # komputasi numerik di pyutk bekerja dgn array multimensional
from io import BytesIO # bekerja dgn data biner
import json # bekerja dgn data dalam format json

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file #kelas utama utk membuat web, merender html, mengakses data dari permintaan http yg masuk, mengarahkan pengguna ke url lain, membangun url, menampilkan pesan satu kali, mengembalikan respons json dari endpoint api, dan mengirimkan file sbg respons http
from flask_sqlalchemy import SQLAlchemy #ORM agar mudah berinteraksi dgn db
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user #utk manajemen sesi, mengelola proses login logout, utk properti dan metode yg dibutuhkan flask-login, utk melakukan login dan logout, dekorate yg memastikan pengguna harus login utk melakukan route, dan objek proxy yg mewakili pengguna yg sedang login 
from werkzeug.security import generate_password_hash, check_password_hash #meng hash pw sebelum menyimpan ke db, memverifikasi pw
from sklearn.preprocessing import StandardScaler, LabelEncoder #z-core dan labelencoder
from sklearn.cluster import KMeans # implementasi algoritma k-means
from sklearn.metrics import silhouette_score #utk mengevasuliasi kulitas hasil klusterisasi
import plotly.express as px #menyediakan antarmuka sederhana dan cepat utk membuat jenis grafik umum
import plotly.graph_objects as go  # utk membuat grafik yg lebih kompleks
from sqlalchemy import func #utk menggunakan fungsi agregat db seperti count, sum, avg

from reportlab.lib.pagesizes import A4 # menentukan halaman pdf
from reportlab.lib.units import inch # mengukur dimensi inci
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image #membuat pdf, menambah paragraf, menambah spaci vertikal kosong, membuat tabel dan menerapkan style table, memulai halaman baru, dan menyisipkan gambar
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  #mutk mendapatkan styleheet bawaan, dan membuat style paragraf
from reportlab.lib import colors #menggunakan definisi warna standard

# inisialisasi flask k-means dan database
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secret_key_here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'

# --- Models ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    nama_lengkap = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    role = db.Column(db.String(50), nullable=False, default='Kancab')
    cabang = db.Column(db.String(100), nullable=True)
    def __repr__(self):
        return f'<User {self.username} - {self.role}>'
    def get_id(self):
        return str(self.id)
    @property
    def is_active(self):
        return True
    @property
    def is_authenticated(self):
        return True
    @property
    def is_anonymous(self):
        return False

class Cluster(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    k_value = db.Column(db.Integer, nullable=False)
    silhouette_score_value = db.Column(db.Float, nullable=True)
    features_used = db.Column(db.String(500), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    cluster_characteristics_json = db.Column(db.Text, nullable=True)
    plot_json = db.Column(db.Text, nullable=True)
    def __repr__(self):
        return f'<Cluster Run ID={self.id} K={self.k_value} Score={self.silhouette_score_value}>'

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nomor_pensiun = db.Column(db.String(100), nullable=False)
    penerima = db.Column(db.String(255), nullable=False)
    status_pensiun = db.Column(db.String(100))
    cabang = db.Column(db.String(100))
    mitra = db.Column(db.String(100))
    status_auten = db.Column(db.String(100))
    bulan = db.Column(db.Integer)
    usia = db.Column(db.Integer)
    jenis_pekerjaan = db.Column(db.String(100))
    cluster_id = db.Column(db.Integer, nullable=True)
    cluster_name = db.Column(db.String(100), nullable=True)
    waktu_upload = db.Column(db.DateTime, default=datetime.utcnow)
    user_input_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    user_input = db.relationship('User', backref=db.backref('uploaded_data', lazy=True))
    clustering_run_id = db.Column(db.Integer, db.ForeignKey('cluster.id'), nullable=True)
    clustering_run = db.relationship('Cluster', backref=db.backref('clustered_data', lazy=True))
    def __repr__(self):
        return f'<Data {self.nomor_pensiun} - Cluster {self.cluster_id}>'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- Helper Functions ---
def get_clusterable_features():
    features = [
        {'name': 'usia', 'type': 'numerik'},
        {'name': 'status_pensiun', 'type': 'kategorikal'},
        {'name': 'jenis_pekerjaan', 'type': 'kategorikal'},
        {'name': 'mitra', 'type': 'kategorikal'}
    ]
    return features

def get_data_for_clustering(df_source, selected_features):
   # pengolahan data sebelum diklusterisasi, encoding dan scalling
    if df_source.empty:
        return pd.DataFrame(), pd.DataFrame(), None 
    df = df_source.copy()
    if not selected_features:
        flash("Tidak ada fitur yang dipilih untuk klasterisasi.", 'warning')
        return df, pd.DataFrame(), None
    available_features = [f for f in selected_features if f in df.columns]
    if not available_features:
        flash("Fitur yang dipilih tidak ditemukan di data yang diunggah.", 'warning')
        return df, pd.DataFrame(), None
    df_cluster = df[available_features].copy() 
    encoders = {}
    scalers = {}
    for feature in available_features:
     # numerik = scaling
        if feature in ['usia', 'bulan']:
            if df_cluster[feature].isnull().any():
                df_cluster[feature] = df_cluster[feature].fillna(df_cluster[feature].mean())
            scaler = StandardScaler()
            df_cluster[feature] = scaler.fit_transform(df_cluster[[feature]])
            scalers[feature] = scaler 
        # kategorikal = encoding
        else: 
            if df_cluster[feature].isnull().any():
                df_cluster[feature] = df_cluster[feature].fillna('Unknown')
            encoder = LabelEncoder()
            df_cluster[feature] = encoder.fit_transform(df_cluster[feature])
            encoders[feature] = encoder 
    return df, df_cluster, encoders 

def perform_clustering(df, df_cluster, n_clusters):
    # inti proses klasterisasi + evalusasi kualitas klaster
    if df_cluster.empty:
        flash("DataFrame untuk klasterisasi kosong. Tidak dapat melakukan klasterisasi.", 'warning')
        return None, None, None
    try:
        # proses k-means serta prediksi klaster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) 
        clusters = kmeans.fit_predict(df_cluster)
        score = None
        # hitung silhouette score
        if n_clusters > 1 and len(df_cluster) >= n_clusters:
            try:
                score = silhouette_score(df_cluster, clusters)
            except Exception as se:
                flash(f"Error menghitung Silhouette Score: {se}. Pastikan data memiliki variasi yang cukup untuk membentuk klaster yang berbeda.", 'warning')
                score = None 
        else:
            flash("Silhouette Score tidak dapat dihitung (jumlah klaster=1 atau data terlalu sedikit).", 'info')
       # menambhakan hasil klaster ke dataframe
        df['cluster_id'] = clusters
        df['cluster_name'] = df['cluster_id'].apply(lambda x: f'Klaster {x + 1}')
        return df, score, kmeans.cluster_centers_
    except Exception as e:
        flash(f"Error during clustering: {e}", 'danger')
        return None, None, None

def calculate_dashboard_stats(user_cabang=None):
    # summary statistik dashboard
    query = Data.query
    if user_cabang:
        query = query.filter_by(cabang=user_cabang)
    total_data = query.count()
    sudah_auten = query.filter_by(status_auten='Autentikasi Berhasil').count() 
    belum_auten = query.filter_by(status_auten='Belum Autentikasi').count()
    gagal_auten = query.filter_by(status_auten='Gagal Autentikasi').count()
    #hitung distribusi status auten, jenis pekerjaan, status pensiun
    dist_status_auten = query.with_entities(Data.status_auten, func.count(Data.status_auten)).group_by(Data.status_auten).all()
    dist_status_auten_dict = {status: count for status, count in dist_status_auten}
    dist_status_auten_percent = {status: f"{(count / total_data * 100):.2f}" for status, count in dist_status_auten_dict.items()} if total_data > 0 else {}
    dist_jenis_pekerjaan = query.with_entities(Data.jenis_pekerjaan, func.count(Data.jenis_pekerjaan)).group_by(Data.jenis_pekerjaan).all()
    dist_jenis_pekerjaan_dict = {jenis: count for jenis, count in dist_jenis_pekerjaan}
    dist_jenis_pekerjaan_percent = {jenis: f"{(count / total_data * 100):.2f}" for jenis, count in dist_jenis_pekerjaan_dict.items()} if total_data > 0 else {}
    dist_status_pensiun = query.with_entities(Data.status_pensiun, func.count(Data.status_pensiun)).group_by(Data.status_pensiun).all()
    dist_status_pensiun_dict = {status: count for status, count in dist_status_pensiun}
    dist_status_pensiun_percent = {status: f"{(count / total_data * 100):.2f}" for status, count in dist_status_pensiun_dict.items()} if total_data > 0 else {}
    # ambil run klasterisasi terbaru utk ditampilkan di dashboard
    last_cluster_run = Cluster.query.order_by(Cluster.timestamp.desc()).first()
    silhouette_score_display = None
    k_value_display = "N/A"
    tanggal_klasterisasi_display = "Belum ada"
    fitur_digunakan_display = "Tidak Ada"
    if last_cluster_run:
        silhouette_score_display = last_cluster_run.silhouette_score_value
        k_value_display = last_cluster_run.k_value
        tanggal_klasterisasi_display = last_cluster_run.timestamp.strftime('%d-%m-%Y %H:%M:%S')
        fitur_digunakan_display = last_cluster_run.features_used.replace(',', ', ') if last_cluster_run.features_used else "Tidak Ada"
    return {
        'total_data': total_data,
        'sudah_auten': sudah_auten,
        'belum_auten': belum_auten,
        'gagal_auten': gagal_auten,
        'distribusi_status_auten': dist_status_auten_percent,
        'distribusi_jenis_pekerjaan': dist_jenis_pekerjaan_percent,
        'distribusi_status_pensiun': dist_status_pensiun_percent,
        'silhouette_score_display': silhouette_score_display,
        'k_value_display': k_value_display,
        'tanggal_klasterisasi_display': tanggal_klasterisasi_display,
        'fitur_digunakan_display': fitur_digunakan_display
    }

# --- FUNGSI DINAMIS NARASI & REKOMENDASI KLASTER ---
def narasi_klaster(cluster_detail):
    #membuat narasi dan rerkomendasi utk karakteristik setiap klaster
    usia_avg = cluster_detail.get('rata_rata_usia', '-')
    pekerjaan = cluster_detail.get('distribusi_jenis_pekerjaan', {})
    pekerjaan_utama = max(pekerjaan, key=pekerjaan.get) if pekerjaan else '-'
    status = cluster_detail.get('distribusi_status_pensiun', {})
    status_utama = max(status, key=status.get) if status else '-'
    auten = cluster_detail.get('distribusi_status_auten', {})
    auten_utama = max(auten, key=auten.get) if auten else '-'
    rekom = []
    narasi = f"Klaster ini didominasi oleh pensiunan dengan rata-rata usia {usia_avg:.2f} tahun, mayoritas berstatus pensiun {status_utama} dan memiliki jenis pekerjaan utama {pekerjaan_utama}. Status autentikasi terbanyak adalah {auten_utama}."
    if auten_utama.lower() == 'belum autentikasi':
        rekom.append("Perlu dilakukan sosialisasi atau pendampingan autentikasi untuk kelompok ini.")
    if isinstance(usia_avg, (int, float)) and usia_avg < 55:
        rekom.append("Kelompok ini didominasi usia muda, perlu monitoring lebih lanjut terkait pola klaim.")
    if pekerjaan_utama.lower() == 'polri':
        rekom.append("Klaster ini banyak diisi oleh pensiunan Polri, dapat dianalisis lebih lanjut terkait kebutuhan spesifik.")
    return narasi, rekom

@app.route('/')
@login_required
def index():
    # routing utama sesuai role
    if current_user.role.lower() == 'admin':
        return redirect(url_for('dashboard_admin'))
    elif current_user.role.lower() == 'kancab':
        return redirect(url_for('dashboard_kancab'))
    else:
        flash("Peran pengguna tidak dikenali.", 'danger')
        logout_user()
        return redirect(url_for('login_page'))

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    # login page utk user
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login berhasil!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Username atau password salah.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Anda telah berhasil logout.', 'info')
    return redirect(url_for('login_page'))

@app.route('/dashboard/admin')
@login_required
def dashboard_admin():
    if current_user.role.lower() != 'admin': 
        flash('Anda tidak memiliki akses ke halaman ini.', 'danger')
        return redirect(url_for('dashboard_kancab')) 

    stats = calculate_dashboard_stats() # Tanpa filter cabang
    # plot distribusi status auten
    status_auten_data = db.session.query(Data.status_auten, func.count(Data.id)).group_by(Data.status_auten).all()
    df_status_auten = pd.DataFrame(status_auten_data, columns=['Status Autentikasi', 'Jumlah'])
    plot_status_auten_div = ""
    if not df_status_auten.empty:
        fig_auten = px.pie(df_status_auten, names='Status Autentikasi', values='Jumlah', title='Distribusi Status Autentikasi')
        plot_status_auten_div = fig_auten.to_html(full_html=False)
    else:
        plot_status_auten_div = "<p class='text-center text-muted'>Tidak ada data status autentikasi untuk ditampilkan.</p>"
    # plot distribusi usia
    usia_data = Data.query.with_entities(Data.usia).all()
    df_usia = pd.DataFrame(usia_data, columns=['Usia'])
    plot_usia_div = ""
    if not df_usia.empty and 'Usia' in df_usia.columns and df_usia['Usia'].dropna().any():
        fig_usia = px.histogram(df_usia, x='Usia', nbins=10, title='Distribusi Usia Pensiunan')
        plot_usia_div = fig_usia.to_html(full_html=False)
    else:
        plot_usia_div = "<p class='text-center text-muted'>Tidak ada data usia untuk ditampilkan.</p>"


    return render_template('dashboard_admin.html',
                           stats=stats, 
                           plot_status_auten_div=plot_status_auten_div,
                           plot_usia_div=plot_usia_div
                           )

@app.route('/dashboard/kancab')
@login_required
def dashboard_kancab():
    if current_user.role.lower() != 'kancab': 
        flash('Anda tidak memiliki akses ke halaman ini.', 'danger')
        return redirect(url_for('dashboard_admin')) 

    user_cabang = current_user.cabang # Ambil cabang dari user yang login
    if not user_cabang:
        flash("Profil pengguna Anda tidak memiliki informasi cabang. Harap hubungi Admin.", 'warning')
        # Return a basic dashboard or redirect
        return render_template('dashboard_kancab.html', stats=calculate_dashboard_stats(), user_cabang=None)

    stats = calculate_dashboard_stats(user_cabang=user_cabang) # Filter statistik berdasarkan cabang

    # Query data yang difilter berdasarkan cabang untuk plot
    filtered_data_query = Data.query.filter_by(cabang=user_cabang)
    
    # Plot Distribusi Status Autentikasi (difilter cabang)
    status_auten_data = filtered_data_query.with_entities(Data.status_auten, func.count(Data.id)).group_by(Data.status_auten).all()
    df_status_auten = pd.DataFrame(status_auten_data, columns=['Status Autentikasi', 'Jumlah'])
    plot_status_auten_div = ""
    if not df_status_auten.empty:
        fig_auten = px.pie(df_status_auten, names='Status Autentikasi', values='Jumlah', title=f'Distribusi Status Autentikasi Cabang {user_cabang}')
        plot_status_auten_div = fig_auten.to_html(full_html=False)
    else:
        plot_status_auten_div = "<p class='text-center text-muted'>Tidak ada data status autentikasi untuk ditampilkan di cabang ini.</p>"


    # Plot Distribusi Jenis Pekerjaan (difilter cabang)
    jenis_pekerjaan_data = filtered_data_query.with_entities(Data.jenis_pekerjaan, func.count(Data.id)).group_by(Data.jenis_pekerjaan).all()
    df_jenis_pekerjaan = pd.DataFrame(jenis_pekerjaan_data, columns=['Jenis Pekerjaan', 'Jumlah'])
    plot_jenis_pekerjaan_div = ""
    if not df_jenis_pekerjaan.empty:
        fig_jenis_pekerjaan = px.bar(df_jenis_pekerjaan, x='Jenis Pekerjaan', y='Jumlah', title=f'Distribusi Jenis Pekerjaan Cabang {user_cabang}')
        plot_jenis_pekerjaan_div = fig_jenis_pekerjaan.to_html(full_html=False)
    else:
        plot_jenis_pekerjaan_div = "<p class='text-center text-muted'>Tidak ada data jenis pekerjaan untuk ditampilkan di cabang ini.</p>"


    # Plot Distribusi Status Pensiun (difilter cabang)
    status_pensiun_data = filtered_data_query.with_entities(Data.status_pensiun, func.count(Data.id)).group_by(Data.status_pensiun).all()
    df_status_pensiun = pd.DataFrame(status_pensiun_data, columns=['Status Pensiun', 'Jumlah'])
    plot_status_pensiun_div = ""
    if not df_status_pensiun.empty:
        fig_status_pensiun = px.bar(df_status_pensiun, x='Status Pensiun', y='Jumlah', title=f'Distribusi Status Pensiun Cabang {user_cabang}')
        plot_status_pensiun_div = fig_status_pensiun.to_html(full_html=False)
    else:
        plot_status_pensiun_div = "<p class='text-center text-muted'>Tidak ada data distribusi status pensiun untuk ditampilkan di cabang ini.</p>"

    # Data untuk "Pensiunan yang Perlu Perhatian" (contoh: Belum Autentikasi di cabang ini)
    pensiunan_perlu_perhatian = filtered_data_query.filter_by(status_auten='Belum Autentikasi').limit(10).all()


    return render_template('dashboard_kancab.html', 
                           stats=stats,
                           plot_status_auten_div=plot_status_auten_div,
                           plot_jenis_pekerjaan_div=plot_jenis_pekerjaan_div,
                           plot_status_pensiun_div=plot_status_pensiun_div,
                           user_cabang=user_cabang,
                           pensiunan_perlu_perhatian=pensiunan_perlu_perhatian)


@app.route('/users')
@login_required
def users_page():
    if current_user.role.lower() != 'admin': 
        flash('Anda tidak memiliki akses ke halaman ini.', 'danger')
        return redirect(url_for('index'))
    
    users = User.query.all()
    return render_template('manage_users.html', users=users)

@app.route('/add_user', methods=['POST'])
@login_required
def add_user():
    if current_user.role.lower() != 'admin': 
        flash('Anda tidak memiliki izin untuk menambah pengguna.', 'danger')
        return redirect(url_for('users_page'))

    username = request.form['username']
    password = request.form['password']
    nama_lengkap = request.form['nama_lengkap'] 
    email = request.form['email']
    role = request.form['role']
    
    cabang_for_kancab = None 
    # Jika role adalah Kancab, set role Cabang
    if role.lower() == 'kancab':
        pass 
    if User.query.filter_by(username=username).first():
        flash(f'Username {username} sudah ada.', 'danger')
        return redirect(url_for('users_page'))
    if User.query.filter_by(email=email).first():
        flash(f'Email {email} sudah terdaftar.', 'danger')
        return redirect(url_for('users_page'))


    hashed_password = generate_password_hash(password, method='scrypt') 
    
    try:
        new_user = User(username=username, password=hashed_password, nama_lengkap=nama_lengkap, email=email, role=role, cabang=cabang_for_kancab)
        db.session.add(new_user)
        db.session.commit()
        flash(f'Pengguna {username} berhasil ditambahkan!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Gagal menambahkan pengguna: {e}', 'danger')
    
    return redirect(url_for('users_page'))

@app.route('/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role.lower() != 'admin': 
        flash('Anda tidak memiliki izin untuk menghapus pengguna.', 'danger')
        return redirect(url_for('users_page'))

    user_to_delete = User.query.get_or_404(user_id)
    
    if user_to_delete.role.lower() == 'admin' and user_to_delete.id == current_user.id: 
        flash('Anda tidak dapat menghapus akun Admin Anda sendiri.', 'danger')
    else:
        try:
            db.session.delete(user_to_delete)
            db.session.commit()
            flash(f'Pengguna {user_to_delete.username} berhasil dihapus.', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Gagal menghapus pengguna: {e}', 'danger')
    
    return redirect(url_for('users_page'))

@app.route('/clustering', methods=['GET', 'POST'])
@login_required
def clustering_page():
    # upload dan proses klasterisasi excel
    num_clusters_display = request.form.get('num_clusters', 3, type=int) 
    selected_features = request.form.getlist('features') 

    plot_data_json = "" 
    table_preview_html = ""
    stats = calculate_dashboard_stats() 
    cluster_details = [] 

    if request.method == 'POST' and current_user.role.lower() == 'admin':
    # validasi file unggah
        if 'file' not in request.files:
            flash('Tidak ada file yang diunggah.', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('Tidak ada file yang dipilih.', 'danger')
            return redirect(request.url)

        if file:
            try:
               # normalisasi kolom 
                df_excel = pd.read_excel(file)
                df_excel.columns = df_excel.columns.str.lower().str.replace(' ', '_')
                print("\n--- DEBUG: df_excel columns after initial normalization ---")
                print(df_excel.columns)
                print("---------------------------------------------------\n")
                # --- PENANGANAN NAMA KOLOM ALTERNATIF ---
                if 'usia_' in df_excel.columns and 'usia' not in df_excel.columns:
                    df_excel.rename(columns={'usia_': 'usia'}, inplace=True)
                    print("--- DEBUG: Renamed 'usia_' to 'usia' ---")
                
                if 'status_autentikasi' in df_excel.columns and 'status_auten' not in df_excel.columns:
                    df_excel.rename(columns={'status_autentikasi': 'status_auten'}, inplace=True)
                    print("--- DEBUG: Renamed 'status_autentikasi' to 'status_auten' ---")
                
                print("\n--- DEBUG: df_excel columns after alternative name handling ---")
                print(df_excel.columns)
                print("---------------------------------------------------\n")
                # --- VALIDASI KOLOM WAJIB MINIMUM ---
                minimum_excel_columns = [
                    'nomor_pensiun', 'penerima', 'status_pensiun', 
                    'mitra', 'usia', 'jenis_pekerjaan'
                ]
                missing_minimum_columns = [col for col in minimum_excel_columns if col not in df_excel.columns]
                if missing_minimum_columns:
                    flash(f"File Excel tidak memiliki kolom wajib minimum: {', '.join([c.replace('_', ' ').title() for c in missing_minimum_columns])}. Harap periksa format file Anda.", 'danger')
                    return render_template('clustering.html',
                                           num_clusters_display=num_clusters_display,
                                           clusterable_features=get_clusterable_features(),
                                           selected_features=selected_features,
                                           stats=stats,
                                           plot_data_json=plot_data_json,
                                           table_preview_html=table_preview_html,
                                           current_user_role=current_user.role.lower())
                # --- PRA-PEMROSESAN DAN PENGISIAN NILAI HILANG ---
                if 'status_auten' not in df_excel.columns:
                    df_excel['status_auten'] = 'Belum Autentikasi'
                    print("--- DEBUG: 'status_auten' column added with default 'Belum Autentikasi' ---")
                else:
                    df_excel['status_auten'] = df_excel['status_auten'].fillna('Belum Autentikasi') 
                
                if 'cabang' not in df_excel.columns: # Handle 'cabang' if missing
                    df_excel['cabang'] = 'Tidak Diketahui'
                else:
                    df_excel['cabang'] = df_excel['cabang'].fillna('Tidak Diketahui')

                df_excel['status_pensiun'] = df_excel['status_pensiun'].fillna('Tidak Diketahui') 
                df_excel['jenis_pekerjaan'] = df_excel['jenis_pekerjaan'].fillna('Tidak Diketahui') 
                df_excel['mitra'] = df_excel['mitra'].fillna('Tidak Diketahui') 
                
                if 'bulan' in df_excel.columns:
                    df_excel['bulan'] = pd.to_numeric(df_excel['bulan'], errors='coerce').fillna(0).astype(int)
                else:
                    df_excel['bulan'] = 0 

                df_excel['usia'] = pd.to_numeric(df_excel['usia'], errors='coerce').fillna(0).astype(int)

                # --- Lakukan Klasterisasi ---
                # Pass only the relevant features for clustering to get_data_for_clustering
                df_all_data, df_cluster_processed, encoders = get_data_for_clustering(df_excel, selected_features)

                if not df_all_data.empty and not df_cluster_processed.empty and selected_features:
                    df_clustered, silhouette_score_val, _ = perform_clustering(df_all_data.copy(), df_cluster_processed, num_clusters_display)
                    
                    if df_clustered is not None:
                        # Generate plot JSON for this run
                        df_to_plot = df_clustered.copy()
                        if len(selected_features) >= 2:
                            feature1 = selected_features[0]
                            feature2 = selected_features[1]
                            feature_types = {f['name']: f['type'] for f in get_clusterable_features()}
                            type1 = feature_types.get(feature1)
                            type2 = feature_types.get(feature2)

                            if feature1 in df_to_plot.columns and feature2 in df_to_plot.columns:
                                has_variation_f1 = df_to_plot[feature1].nunique() > 1
                                has_variation_f2 = df_to_plot[feature2].nunique() > 1

                                if has_variation_f1 and has_variation_f2:
                                    if type1 == 'numerik' and type2 == 'numerik':
                                        fig = px.scatter(df_to_plot, x=feature1, y=feature2, color='cluster_name',
                                                        title=f'Klasterisasi Data Pensiun ({feature1.replace("_", " ").title()} vs {feature2.replace("_", " ").title()})',
                                                        hover_data=selected_features + ['cluster_name'],
                                                        height=500)
                                        plot_data_json = fig.to_json()
                                    elif (type1 == 'numerik' and type2 == 'kategorikal') or \
                                         (type1 == 'kategorikal' and type2 == 'numerik'):
                                        if type1 == 'numerik':
                                            x_axis = feature2
                                            y_axis = feature1
                                        else:
                                            x_axis = feature1
                                            y_axis = feature2
                                        fig = px.box(df_to_plot, x=x_axis, y=y_axis, color='cluster_name',
                                                     title=f'Klasterisasi Data Pensiun ({y_axis.replace("_", " ").title()} per {x_axis.replace("_", " ").title()})',
                                                     hover_data=selected_features + ['cluster_name'],
                                                     height=500)
                                        plot_data_json = fig.to_json()
                                    else: # Both are categorical
                                        fig = px.scatter(df_to_plot, x=feature1, y=feature2, color='cluster_name',
                                                        title=f'Klasterisasi Data Pensiun ({feature1.replace("_", " ").title()} vs {feature2.replace("_", " ").title()})',
                                                        hover_data=selected_features + ['cluster_name'],
                                                        height=500)
                                        plot_data_json = fig.to_json()
                                else:
                                    plot_data_json = "" # No meaningful plot if no variation
                            else:
                                plot_data_json = "" # No meaningful plot if features not found
                        elif len(selected_features) == 1:
                            feature1 = selected_features[0]
                            if feature1 in df_to_plot.columns and df_to_plot[feature1].nunique() > 1:
                                fig = px.histogram(df_to_plot, x=feature1, color='cluster_name',
                                                   title=f'Distribusi {feature1.replace("_", " ").title()} per Klaster',
                                                   hover_data=[feature1, 'cluster_name'],
                                                   height=500)
                                plot_data_json = fig.to_json()
                            else:
                                plot_data_json = ""
                        else:
                            plot_data_json = "" # No features selected for plot

                        # --- HITUNG DETAIL PER KLASTER ---
                        cluster_details = []
                        for i in range(num_clusters_display):
                            cluster_df = df_clustered[df_clustered['cluster_id'] == i]
                            
                            detail = {
                                'id': i,
                                'name': f'Klaster {i + 1}',
                                'total_data': len(cluster_df)
                            }

                            if 'usia' in cluster_df.columns:
                                detail['rata_rata_usia'] = cluster_df['usia'].mean()
                            else:
                                detail['rata_rata_usia'] = 'N/A'

                            if 'bulan' in cluster_df.columns:
                                detail['rata_rata_bulan'] = cluster_df['bulan'].mean()
                            else:
                                detail['rata_rata_bulan'] = 'N/A'
                            
                            # Categorical features for details - now only those that are not removed from clustering
                            categorical_features_for_details = ['status_pensiun', 'jenis_pekerjaan', 'mitra', 'status_auten', 'cabang'] # All original columns
                            for cat_feature in categorical_features_for_details:
                                if cat_feature in cluster_df.columns:
                                    distribution = cluster_df[cat_feature].value_counts().to_dict()
                                    detail[f'distribusi_{cat_feature}'] = distribution
                                else:
                                    detail[f'distribusi_{cat_feature}'] = {}

                            cluster_details.append(detail) 
                        # --- AKHIR HITUNG DETAIL PER KLASTER ---

                        # --- SIMPAN HASIL KLASTERISASI KE TABEL CLUSTER (HISTORIS) ---
                        new_cluster_run = Cluster(
                            k_value=num_clusters_display,
                            silhouette_score_value=silhouette_score_val,
                            features_used=",".join(selected_features),
                            cluster_characteristics_json=json.dumps(cluster_details), # Simpan detail klaster sebagai JSON
                            plot_json=plot_data_json # Simpan plot data sebagai JSON
                        )
                        db.session.add(new_cluster_run)
                        db.session.commit()

                        # --- Hapus semua data lama dari tabel Data sebelum menyimpan yang baru ---
                        Data.query.delete()
                        db.session.commit()
                        flash("Data lama berhasil dihapus sebelum mengunggah yang baru.", 'info')

                        data_to_save = []
                        for index, row in df_clustered.iterrows():
                            try:
                                new_data_item = Data(
                                    nomor_pensiun=str(row['nomor_pensiun']),
                                    penerima=str(row['penerima']),
                                    status_pensiun=str(row['status_pensiun']),
                                    cabang=str(row['cabang']), # Tetap simpan cabang
                                    mitra=str(row['mitra']),
                                    status_auten=str(row['status_auten']), 
                                    bulan=int(row['bulan']), # Tetap simpan bulan
                                    usia=int(row['usia']), 
                                    jenis_pekerjaan=str(row['jenis_pekerjaan']),
                                    cluster_id=int(row['cluster_id']),
                                    cluster_name=str(row['cluster_name']),
                                    waktu_upload=datetime.utcnow(),
                                    user_input_id=current_user.id,
                                    clustering_run_id=new_cluster_run.id # Link to the new clustering run
                                )
                                data_to_save.append(new_data_item)
                            except Exception as ex:
                                nomor_pensiun_debug = row.get('nomor_pensiun', 'N/A')
                                flash(f"Error memproses baris {index+2} (Nomor Pensiun: {nomor_pensiun_debug}): {ex}. Baris ini dilewati.", 'warning')
                                continue
                        db.session.add_all(data_to_save)
                        db.session.commit()
                        flash('Data Excel berhasil diunggah, diklasterisasi, dan disimpan.', 'success')
                    else:
                        flash("Gagal mendapatkan hasil klasterisasi. Pastikan data memiliki variasi yang cukup.", 'danger')
                else:
                    flash("Tidak ada data atau fitur yang valid untuk klasterisasi setelah unggah.", 'warning')

            except pd.errors.EmptyDataError:
                flash('File Excel kosong atau tidak valid.', 'danger')
            except FileNotFoundError:
                flash('File tidak ditemukan (internal error).', 'danger')
            except Exception as e:
                flash(f'Terjadi kesalahan saat memproses file: {e}', 'danger')

# ambil hasil run utk preview
    last_cluster_run_obj = Cluster.query.order_by(Cluster.timestamp.desc()).first()
    if last_cluster_run_obj:
        plot_data_json = last_cluster_run_obj.plot_json if last_cluster_run_obj.plot_json else ""
        cluster_details = json.loads(last_cluster_run_obj.cluster_characteristics_json) if last_cluster_run_obj.cluster_characteristics_json else []
        
        latest_data_for_preview = Data.query.filter_by(clustering_run_id=last_cluster_run_obj.id).limit(10).all()
        if latest_data_for_preview:
            df_preview = pd.DataFrame([row.__dict__ for row in latest_data_for_preview])
            df_preview = df_preview.drop(columns=['_sa_instance_state'], errors='ignore')
            display_cols = [col for col in ['nomor_pensiun', 'penerima', 'usia', 'status_auten', 'cluster_name'] if col in df_preview.columns]
            table_preview_html = df_preview[display_cols].to_html(classes='table table-bordered table-hover', index=False)
        else:
            table_preview_html = "<p class='text-muted'>Belum ada data klasterisasi untuk ditampilkan.</p>"
    else:
        plot_data_json = ""
        cluster_details = []
        table_preview_html = "<p class='text-muted'>Belum ada data klasterisasi untuk ditampilkan. Silakan unggah dan jalankan klasterisasi sebagai Admin.</p>"

    clusterable_features = get_clusterable_features()
    stats = calculate_dashboard_stats()

    return render_template('clustering.html',
                           num_clusters_display=num_clusters_display,
                           clusterable_features=clusterable_features,
                           selected_features=selected_features,
                           stats=stats,
                           plot_data_json=plot_data_json, 
                           table_preview_html=table_preview_html,
                           cluster_details=cluster_details, 
                           current_user_role=current_user.role.lower()) 

@app.route('/reports')
@login_required
def reports_page():
    # Filter data berdasarkan cabang jika pengguna adalah Kancab
    query = Data.query
    if current_user.role.lower() == 'kancab' and current_user.cabang:
        query = query.filter_by(cabang=current_user.cabang)

    latest_clustered_data = query.all()

    # Historical runs remain global for now, Kancab can see global history but filtered data
    historical_runs = Cluster.query.order_by(Cluster.timestamp.desc()).all()
    
    stats = calculate_dashboard_stats()

    return render_template('reports.html', 
                           stats=stats, 
                           historical_runs=historical_runs,
                           all_data=latest_clustered_data,
                           current_user_role=current_user.role.lower(),
                           user_cabang=current_user.cabang)

@app.route('/report_detail/<string:nomor_pensiun>')
@login_required
def report_detail_page(nomor_pensiun):
    # query data pensiun berdasarkan nopen
    data_item_query = Data.query.filter_by(nomor_pensiun=nomor_pensiun)
    if current_user.role.lower() == 'kancab' and current_user.cabang:
        data_item_query = data_item_query.filter_by(cabang=current_user.cabang)
    data_item = data_item_query.first_or_404()
    current_run_id = data_item.clustering_run_id
    # sbmil objek cluster run terkait data ini
    current_cluster_run_obj = Cluster.query.get(current_run_id)
    cluster_details_global = []
    plot_data_json_global = ""
    # detail klaster dan plot json
    if current_cluster_run_obj:
        plot_data_json_global = current_cluster_run_obj.plot_json if current_cluster_run_obj.plot_json else ""
        cluster_details_global = json.loads(current_cluster_run_obj.cluster_characteristics_json) if current_cluster_run_obj.cluster_characteristics_json else []
    selected_cluster_detail = next((item for item in cluster_details_global if item['id'] == data_item.cluster_id), None)
    narasi = ""
    rekomendasi = []
    # generate narasi dan rekomendasi
    if selected_cluster_detail:
        narasi, rekomendasi = narasi_klaster(selected_cluster_detail)
    # render detail dgn data, narasi, rekomendasi, dan visualisasi
    return render_template('report_detail.html', 
                           data_item=data_item, 
                           cluster_details_global=cluster_details_global,
                           selected_cluster_detail=selected_cluster_detail,
                           plot_data_json_global=plot_data_json_global,
                           narasi_klaster=narasi,
                           rekomendasi_klaster=rekomendasi)

# --- PATCH: PDF GENERATOR DENGAN NARASI DINAMIS (report lab)---
def generate_global_pdf(historical_run_id):
    historical_run = Cluster.query.get(historical_run_id)
    if not historical_run:
        return None

    # detail klaster dan plot
    cluster_details = json.loads(historical_run.cluster_characteristics_json) if historical_run.cluster_characteristics_json else []
    plot_data_json = historical_run.plot_json if historical_run.plot_json else ""

    buffer = BytesIO()
    # inisialisasi dokumen pdf
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=inch/2, leftMargin=inch/2,
                            topMargin=inch/2, bottomMargin=inch/2)
    # style custom bagian laporan
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitleStyle', parent=styles['h1'], fontSize=20, leading=24, alignment=1, spaceAfter=12, textColor=colors.blue))
    styles.add(ParagraphStyle(name='CustomSubtitleStyle', parent=styles['h2'], fontSize=14, leading=18, alignment=1, spaceAfter=12, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name='CustomHeading1', parent=styles['h1'], fontSize=16, leading=18, spaceAfter=10, textColor=colors.darkgreen))
    styles.add(ParagraphStyle(name='CustomHeading2', parent=styles['h2'], fontSize=14, leading=16, spaceAfter=8, textColor=colors.darkcyan))
    styles.add(ParagraphStyle(name='CustomNormal', parent=styles['Normal'], fontSize=10, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='CustomSmall', parent=styles['Normal'], fontSize=8, leading=10, spaceAfter=4, textColor=colors.gray))

    story = []

    # Judul Laporan
    story.append(Paragraph("Laporan Analisis Pola Klaim Peserta", styles['CustomTitleStyle']))
    story.append(Paragraph("PT ASABRI (Persero) Kancab Batam", styles['CustomSubtitleStyle']))
    story.append(Paragraph("Implementasi K-Means Clustering untuk Efisiensi Pengelolaan Klaim dan Pengambilan Keputusan Berbasis Data", styles['CustomNormal']))
    story.append(Paragraph(f"ID Klasterisasi: {historical_run.id}", styles['CustomSmall']))
    story.append(Paragraph(f"Tanggal Laporan: {historical_run.timestamp.strftime('%d %B %Y %H:%M:%S')}", styles['CustomSmall']))
    story.append(Spacer(1, 0.2 * inch))

    # Pendahuluan
    story.append(Paragraph("1. Pendahuluan", styles['CustomHeading1']))
    story.append(Paragraph("Penelitian ini bertujuan untuk mengimplementasikan sistem identifikasi pola klaim peserta PT ASABRI (Persero) Kancab Batam menggunakan algoritma K-Means Clustering, sebagai upaya untuk meningkatkan efisiensi pengelolaan klaim dan mendukung pengambilan keputusan berbasis data.", styles['CustomNormal']))
    story.append(Paragraph("Identifikasi pola klaim sangat penting untuk memahami karakteristik pensiunan dan jenis klaim yang mereka ajukan. Dengan mengelompokkan pensiunan berdasarkan fitur-fitur relevan, PT ASABRI dapat merancang strategi pengelolaan klaim yang lebih efisien, mengidentifikasi potensi risiko atau kebutuhan khusus, serta membuat keputusan bisnis yang lebih tepat berdasarkan data.", styles['CustomNormal']))
    story.append(Spacer(1, 0.2 * inch))

    # Metodologi
    story.append(Paragraph("2. Metodologi Klasterisasi K-Means", styles['CustomHeading1']))
    story.append(Paragraph("Algoritma K-Means adalah metode klasterisasi non-hierarkis yang bertujuan untuk mempartisi n observasi ke dalam k klaster, di mana setiap observasi termasuk dalam klaster dengan rata-rata terdekat (pusat klaster atau centroid). Proses ini melibatkan iterasi untuk meminimalkan jumlah kuadrat jarak antara titik data dan centroid klaster mereka.", styles['CustomNormal']))
    story.append(Paragraph("Fitur-fitur yang digunakan dalam proses klasterisasi ini meliputi:", styles['CustomNormal']))
    
    features_list_data = []
    for feature_name in historical_run.features_used.split(','):
        feature_type = 'Numerik' if feature_name in ['usia', 'bulan'] else 'Kategorikal'
        features_list_data.append([Paragraph(f"• <b>{feature_name.replace('_', ' ').title()}</b>: {feature_type}.", styles['CustomNormal'])])
    
    if features_list_data:
        features_table = Table(features_list_data, colWidths=[6.5*inch])
        features_table.setStyle(TableStyle([
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(features_table)
    story.append(Paragraph(f"Jumlah klaster (K) yang dipilih untuk analisis ini adalah: <b>{historical_run.k_value}</b>.", styles['CustomNormal']))
    story.append(Spacer(1, 0.2 * inch))

    # Hasil dan Analisis
    story.append(Paragraph("3. Hasil dan Analisis Klasterisasi", styles['CustomHeading1']))
    story.append(Paragraph("3.1. Kualitas Klaster (Silhouette Score)", styles['CustomHeading2']))
    story.append(Paragraph("Silhouette Score adalah metrik yang digunakan untuk mengevaluasi kualitas klasterisasi. Nilai Silhouette Score berkisar dari -1 hingga 1, di mana nilai yang lebih tinggi menunjukkan klaster yang lebih baik (lebih padat dan terpisah dengan baik).", styles['CustomNormal']))
    story.append(Paragraph(f"Hasil Silhouette Score untuk klasterisasi ini adalah: <b>{historical_run.silhouette_score_value:.2f}</b>.", styles['CustomNormal']))
    story.append(Paragraph("Nilai ini mengindikasikan bahwa klaster-klaster yang terbentuk cukup kohesif dan terpisah satu sama lain.", styles['CustomNormal'])) # Generic interpretation
    story.append(Spacer(1, 0.2 * inch))

    # Visualisasi Klaster
    story.append(Paragraph("3.2. Visualisasi Klaster", styles['CustomHeading2']))
    story.append(Paragraph("Visualisasi di bawah ini menampilkan hasil klasterisasi dari run ID {}. Plot ini disajikan dalam bentuk Box Plot atau Scatter Plot, diwarnai berdasarkan klaster.".format(historical_run.id), styles['CustomNormal']))
    
    if plot_data_json:
        try:
            fig = go.Figure(json.loads(plot_data_json))
            img_buffer = BytesIO()
            fig.write_image(img_buffer, format='png', width=800, height=500, scale=1)
            img_buffer.seek(0)
            
            img = Image(img_buffer)
            img.drawWidth = 6 * inch
            img.drawHeight = 4 * inch
            story.append(img)
            story.append(Paragraph("Interpretasi Visual: Plot ini membantu memahami bagaimana fitur-fitur pensiunan bervariasi di antara klaster. Perbedaan dalam distribusi antar klaster menunjukkan karakteristik demografi yang berbeda dalam setiap kelompok yang teridentifikasi.", styles['CustomSmall']))
        except Exception as e:
            story.append(Paragraph(f"Gagal memuat visualisasi klaster: {e}", styles['CustomNormal']))
    else:
        story.append(Paragraph("Visualisasi klaster untuk run ini tidak tersedia atau tidak ada variasi data yang cukup untuk plot.", styles['CustomNormal']))
    story.append(Spacer(1, 0.2 * inch))

    # Profil klaster
    story.append(Paragraph("3.3. Profil Setiap Klaster", styles['CustomHeading2']))
    if cluster_details:
        for cluster in cluster_details:
            story.append(Paragraph(f"<b>{cluster['name']}</b> (Total Data: {cluster['total_data']})", styles['CustomNormal']))
            story.append(Paragraph(f"Rata-rata Usia: {cluster['rata_rata_usia']:.2f}" if isinstance(cluster['rata_rata_usia'], (int, float)) else f"Rata-rata Usia: {cluster['rata_rata_usia']}", styles['CustomNormal']))
            story.append(Paragraph(f"Rata-rata Bulan: {cluster['rata_rata_bulan']:.2f}" if isinstance(cluster['rata_rata_bulan'], (int, float)) else f"Rata-rata Bulan: {cluster['rata_rata_bulan']}", styles['CustomNormal']))
            
            story.append(Paragraph("Distribusi Fitur Kategorikal:", styles['CustomNormal']))
            
            cat_features_table_data = []
            for cat_feature_key, distribution in cluster.items():
                if cat_feature_key.startswith('distribusi_') and distribution:
                    feature_name = cat_feature_key.replace('distribusi_', '').replace('_', ' ').title()
                    cat_features_table_data.append([Paragraph(f"<b>{feature_name}:</b>", styles['CustomNormal'])])
                    for item, count in distribution.items():
                        percentage = (count / cluster['total_data'] * 100) if cluster['total_data'] > 0 else 0
                        cat_features_table_data.append([Paragraph(f"• {item.title()}: {count} data ({percentage:.2f}%)", styles['CustomNormal'])])
            
            if cat_features_table_data:
                cat_table = Table(cat_features_table_data, colWidths=[6.5*inch])
                cat_table.setStyle(TableStyle([
                    ('LEFTPADDING', (0,0), (-1,-1), 0),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ]))
                story.append(cat_table)
            # PATCH: narasi & rekomendasi dinamis
            narasi, rekomendasi = narasi_klaster(cluster)
            story.append(Paragraph(f"Narasi Otomatis: {narasi}", styles['CustomSmall']))
            if rekomendasi:
                for rek in rekomendasi:
                    story.append(Paragraph(f"Rekomendasi: {rek}", styles['CustomSmall']))
            story.append(Spacer(1, 0.1 * inch))
    else:
        story.append(Paragraph("Detail setiap klaster untuk run ini tidak ditemukan.", styles['CustomNormal']))
    story.append(Spacer(1, 0.2 * inch))

    # Kesimpulan dan Rekomendasi
    story.append(Paragraph("4. Kesimpulan dan Rekomendasi", styles['CustomHeading1']))
    story.append(Paragraph("Berdasarkan hasil klasterisasi ini, kami dapat mengidentifikasi beberapa pola klaim peserta PT ASABRI (Persero) Kancab Batam. Pola-pola ini dapat dimanfaatkan untuk:", styles['CustomNormal']))
    story.append(Paragraph("• <b>Peningkatan Efisiensi Pengelolaan Klaim:</b> Dengan memahami karakteristik setiap kelompok pensiunan, PT ASABRI dapat mengoptimalkan proses verifikasi, alokasi sumber daya, dan penyelesaian klaim.", styles['CustomNormal']))
    story.append(Paragraph("• <b>Dukungan Pengambilan Keputusan Berbasis Data:</b> Informasi klasterisasi memberikan wawasan yang mendalam untuk pengembangan strategi layanan, identifikasi segmen pensiunan berisiko tinggi atau membutuhkan perhatian khusus, serta perumusan kebijakan yang lebih tepat sasaran.", styles['CustomNormal']))
    story.append(Paragraph("<b>Rekomendasi Spesifik:</b>", styles['CustomNormal']))
    # PATCH: rekomendasi dinamis per klaster
    for cluster in cluster_details:
        _, rekomendasi = narasi_klaster(cluster)
        for rek in rekomendasi:
            story.append(Paragraph(f"• {rek}", styles['CustomNormal']))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("--- Akhir Laporan ---", styles['CustomSmall']))

    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_detail_pdf(nomor_pensiun):
    data_item_query = Data.query.filter_by(nomor_pensiun=nomor_pensiun)
    if current_user.role.lower() == 'kancab' and current_user.cabang:
        data_item_query = data_item_query.filter_by(cabang=current_user.cabang)
    data_item = data_item_query.first()
    if not data_item:
        return None

    current_cluster_run_obj = Cluster.query.get(data_item.clustering_run_id)
    cluster_details_global = json.loads(current_cluster_run_obj.cluster_characteristics_json) if current_cluster_run_obj and current_cluster_run_obj.cluster_characteristics_json else []
    selected_cluster_detail = next((item for item in cluster_details_global if item['id'] == data_item.cluster_id), None)
    plot_data_json_global = current_cluster_run_obj.plot_json if current_cluster_run_obj and current_cluster_run_obj.plot_json else ""

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=inch/2, leftMargin=inch/2,
                            topMargin=inch/2, bottomMargin=inch/2)
    styles = getSampleStyleSheet()

    # Define custom styles using styles.add() with unique names and parent styles
    styles.add(ParagraphStyle(name='CustomTitleStyle', parent=styles['h1'], fontSize=20, leading=24, alignment=1, spaceAfter=12, textColor=colors.blue))
    styles.add(ParagraphStyle(name='CustomSubtitleStyle', parent=styles['h2'], fontSize=14, leading=18, alignment=1, spaceAfter=12, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name='CustomHeading1', parent=styles['h1'], fontSize=16, leading=18, spaceAfter=10, textColor=colors.darkgreen))
    styles.add(ParagraphStyle(name='CustomHeading2', parent=styles['h2'], fontSize=14, leading=16, spaceAfter=8, textColor=colors.darkcyan))
    styles.add(ParagraphStyle(name='CustomNormal', parent=styles['Normal'], fontSize=10, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='CustomSmall', parent=styles['Normal'], fontSize=8, leading=10, spaceAfter=4, textColor=colors.gray))
    styles.add(ParagraphStyle(name='DataLabel', parent=styles['Normal'], fontName='Helvetica-Bold', spaceAfter=2))
    styles.add(ParagraphStyle(name='DataValue', parent=styles['Normal'], spaceAfter=6))


    story = []

    # Title Section
    story.append(Paragraph("Laporan Detail Data Pensiun", styles['CustomTitleStyle']))
    story.append(Paragraph(f"Nomor Pensiun: {data_item.nomor_pensiun}", styles['CustomSubtitleStyle']))
    story.append(Paragraph("Analisis Klasterisasi", styles['CustomNormal']))
    story.append(Paragraph(f"Tanggal Laporan: {datetime.now().strftime('%d %B %Y')}", styles['CustomSmall']))
    story.append(Spacer(1, 0.2 * inch))

    # 1. Informasi Data Pensiun
    story.append(Paragraph("1. Informasi Data Pensiun", styles['CustomHeading1']))
    
    # Use a Table for better alignment, similar to the HTML screenshot
    info_table_data = [
        [Paragraph("Penerima:", styles['DataLabel']), Paragraph(str(data_item.penerima), styles['DataValue'])],
        [Paragraph("Status Pensiun:", styles['DataLabel']), Paragraph(str(data_item.status_pensiun), styles['DataValue'])],
        [Paragraph("Cabang:", styles['DataLabel']), Paragraph(str(data_item.cabang), styles['DataValue'])],
        [Paragraph("Mitra:", styles['DataLabel']), Paragraph(str(data_item.mitra), styles['DataValue'])],
        [Paragraph("Status Autentikasi:", styles['DataLabel']), Paragraph(str(data_item.status_auten), styles['DataValue'])],
        [Paragraph("Bulan:", styles['DataLabel']), Paragraph(str(data_item.bulan), styles['DataValue'])],
        [Paragraph("Usia:", styles['DataLabel']), Paragraph(str(data_item.usia), styles['DataValue'])],
        [Paragraph("Jenis Pekerjaan:", styles['DataLabel']), Paragraph(str(data_item.jenis_pekerjaan), styles['DataValue'])],
        [Paragraph("Klaster Ditemukan:", styles['DataLabel']), Paragraph(str(data_item.cluster_name if data_item.cluster_name else 'Belum Diklaster'), styles['DataValue'])],
        [Paragraph("Waktu Unggah:", styles['DataLabel']), Paragraph(data_item.waktu_upload.strftime('%d-%m-%Y %H:%M:%S') if data_item.waktu_upload else 'N/A', styles['DataValue'])]
    ]
    
    info_table = Table(info_table_data, colWidths=[2.5*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.2 * inch))

    # 2. Penjelasan Klaster
    story.append(Paragraph("2. Penjelasan Klaster", styles['CustomHeading1']))
    if selected_cluster_detail:
        story.append(Paragraph(f"Data ini termasuk dalam <b>{selected_cluster_detail['name']}</b>, yang memiliki karakteristik sebagai berikut:", styles['CustomNormal']))
        story.append(Paragraph(f"Total Data dalam Klaster ini: {selected_cluster_detail['total_data']}", styles['CustomNormal']))
        story.append(Paragraph(f"Rata-rata Usia dalam Klaster: {selected_cluster_detail['rata_rata_usia']:.2f}" if isinstance(selected_cluster_detail['rata_rata_usia'], (int, float)) else f"Rata-rata Usia dalam Klaster: {selected_cluster_detail['rata_rata_usia']}", styles['CustomNormal']))
        story.append(Paragraph(f"Rata-rata Bulan dalam Klaster: {selected_cluster_detail['rata_rata_bulan']:.2f}" if isinstance(selected_cluster_detail['rata_rata_bulan'], (int, float)) else f"Rata-rata Bulan dalam Klaster: {selected_cluster_detail['rata_rata_bulan']}", styles['CustomNormal']))
        
        story.append(Paragraph("Distribusi Fitur Kategorikal dalam Klaster:", styles['CustomNormal']))
        cat_features_table_data = []
        for cat_feature_key, distribution in selected_cluster_detail.items():
            if cat_feature_key.startswith('distribusi_') and distribution:
                feature_name = cat_feature_key.replace('distribusi_', '').replace('_', ' ').title()
                cat_features_table_data.append([Paragraph(f"<b>{feature_name}:</b>", styles['CustomNormal'])])
                for item, count in distribution.items():
                    percentage = (count / selected_cluster_detail['total_data'] * 100) if selected_cluster_detail['total_data'] > 0 else 0
                    cat_features_table_data.append([Paragraph(f"• {item.title()}: {count} data ({percentage:.2f}%)", styles['CustomNormal'])])
        
        if cat_features_table_data:
            cat_table = Table(cat_features_table_data, colWidths=[6.5*inch])
            cat_table.setStyle(TableStyle([
                ('LEFTPADDING', (0,0), (-1,-1), 0),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]))
            story.append(cat_table)
        # PATCH: narasi & rekomendasi dinamis
        narasi, rekomendasi = narasi_klaster(selected_cluster_detail)
        story.append(Paragraph(f"Narasi Otomatis: {narasi}", styles['CustomSmall']))
        if rekomendasi:
            for rek in rekomendasi:
                story.append(Paragraph(f"Rekomendasi: {rek}", styles['CustomSmall']))
    else:
        story.append(Paragraph("Detail klaster untuk data ini tidak ditemukan. Pastikan klasterisasi telah dijalankan.", styles['CustomNormal']))
    story.append(Spacer(1, 0.2 * inch))

    # Global Cluster Visualization
    story.append(Paragraph("3. Visualisasi Klaster Global", styles['CustomHeading1']))
    story.append(Paragraph(f"Plot di bawah ini adalah visualisasi klasterisasi dari seluruh dataset terakhir yang diunggah. Data pensiun dengan Nomor Pensiun {data_item.nomor_pensiun} ini adalah salah satu titik data yang termasuk dalam klaster yang ditunjukkan oleh warnanya.", styles['CustomNormal']))
    
    if plot_data_json_global:
        try:
            fig = go.Figure(json.loads(plot_data_json_global))
            img_buffer = BytesIO()
            fig.write_image(img_buffer, format='png', width=800, height=500, scale=1)
            img_buffer.seek(0)
            
            img = Image(img_buffer)
            img.drawWidth = 6 * inch
            img.drawHeight = 4 * inch
            story.append(img)
            story.append(Paragraph("Interpretasi Visual: Plot ini menampilkan bagaimana klaster-klaster terbentuk secara keseluruhan. Posisi titik data ini dalam plot menunjukkan hubungan usianya dengan kategori Status Pensiun atau Jenis Pekerjaan, dan warna klaster mengidentifikasi kelompoknya.", styles['CustomSmall']))
        except Exception as e:
            story.append(Paragraph(f"Gagal memuat visualisasi klaster global: {e}", styles['CustomNormal']))
    else:
        story.append(Paragraph("Visualisasi klaster global tidak tersedia atau tidak ada variasi data yang cukup untuk plot.", styles['CustomNormal']))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("--- INI LIVE CODINg---", styles['CustomSmall']))
    story.append(Paragraph("INI LIVE CODING", styles['CustomNormal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route('/view_historical_report/<int:cluster_run_id>')
@login_required
def view_historical_report(cluster_run_id):
    historical_run = Cluster.query.get_or_404(cluster_run_id)
    
    cluster_details = json.loads(historical_run.cluster_characteristics_json) if historical_run.cluster_characteristics_json else []
    plot_data_json = historical_run.plot_json if historical_run.plot_json else ""

    # Filter data for this run by branch if user is Kancab
    data_for_this_run_query = Data.query.filter_by(clustering_run_id=cluster_run_id)
    if current_user.role.lower() == 'kancab' and current_user.cabang:
        data_for_this_run_query = data_for_this_run_query.filter_by(cabang=current_user.cabang)
    
    data_for_this_run = data_for_this_run_query.limit(10).all()
    
    return render_template('historical_report_view.html',
                           historical_run=historical_run,
                           cluster_details=cluster_details,
                           plot_data_json=plot_data_json,
                           data_for_this_run=data_for_this_run,
                           current_user_role=current_user.role.lower(),
                           user_cabang=current_user.cabang)


@app.route('/download_excel_detail/<string:nomor_pensiun>')
@login_required
def download_excel_detail(nomor_pensiun):
    data_item_query = Data.query.filter_by(nomor_pensiun=nomor_pensiun)
    if current_user.role.lower() == 'kancab' and current_user.cabang:
        data_item_query = data_item_query.filter_by(cabang=current_user.cabang)
    data_item = data_item_query.first_or_404()
    
    df_item = pd.DataFrame([data_item.__dict__])
    df_item = df_item.drop(columns=['_sa_instance_state'], errors='ignore')

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df_item.to_excel(writer, index=False, sheet_name=f'Detail Pensiun {nomor_pensiun}')
    writer.close()
    output.seek(0)

    return send_file(output, as_attachment=True, download_name=f'detail_pensiun_{nomor_pensiun}.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/download_excel_historical_cluster_data/<int:cluster_run_id>')
@login_required
def download_excel_historical_cluster_data(cluster_run_id):
    data_for_this_run_query = Data.query.filter_by(clustering_run_id=cluster_run_id)
    if current_user.role.lower() == 'kancab' and current_user.cabang:
        data_for_this_run_query = data_for_this_run_query.filter_by(cabang=current_user.cabang)
    data_for_this_run = data_for_this_run_query.all()

    if not data_for_this_run:
        flash("Tidak ada data untuk diunduh untuk klasterisasi ini.", 'warning')
        return redirect(url_for('view_historical_report', cluster_run_id=cluster_run_id))

    df_data = pd.DataFrame([row.__dict__ for row in data_for_this_run])
    df_data = df_data.drop(columns=['_sa_instance_state'], errors='ignore')

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    writer.book.add_worksheet('Data Klasterisasi')
    df_data.to_excel(writer, index=False, sheet_name='Data Klasterisasi')
    
    historical_run = Cluster.query.get(cluster_run_id)
    if historical_run and historical_run.cluster_characteristics_json:
        cluster_details = json.loads(historical_run.cluster_characteristics_json)
        df_cluster_details = pd.DataFrame(cluster_details)
        flat_details = []
        for c in cluster_details:
            flat_row = {k: v for k, v in c.items() if not k.startswith('distribusi_')}
            for dist_key, dist_val in c.items():
                if dist_key.startswith('distribusi_'):
                    feature_name = dist_key.replace('distribusi_', '')
                    for item, count in dist_val.items():
                        flat_row[f'{feature_name}_{item}'] = count
            flat_details.append(flat_row)
        df_cluster_details_flat = pd.DataFrame(flat_details)
        df_cluster_details_flat.to_excel(writer, index=False, sheet_name='Ringkasan Klaster')

    writer.close()
    output.seek(0)

    return send_file(output, as_attachment=True, download_name=f'laporan_klaster_data_{cluster_run_id}.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# --- PDF Generation Functions using ReportLab ---
def generate_global_pdf(historical_run_id):
    historical_run = Cluster.query.get(historical_run_id)
    if not historical_run:
        return None

    cluster_details = json.loads(historical_run.cluster_characteristics_json) if historical_run.cluster_characteristics_json else []
    plot_data_json = historical_run.plot_json if historical_run.plot_json else ""

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=inch/2, leftMargin=inch/2,
                            topMargin=inch/2, bottomMargin=inch/2)

    # Times New Roman font, use built-in 'Times-Roman' for ReportLab
    times_font = 'Times-Roman'
    times_font_bold = 'Times-Bold'
    times_font_italic = 'Times-Italic'

    styles = getSampleStyleSheet()
    # Overwrite styles for Times New Roman (Times-Roman in ReportLab)
    styles.add(ParagraphStyle(name='CustomTitleStyle', parent=styles['h1'], fontName=times_font_bold, fontSize=20, alignment=1, spaceAfter=12, textColor=colors.black))
    styles.add(ParagraphStyle(name='CustomSubtitleStyle', parent=styles['h2'], fontName=times_font_bold, fontSize=14, alignment=1, spaceAfter=12, textColor=colors.black))
    styles.add(ParagraphStyle(name='CustomHeading1', parent=styles['h1'], fontName=times_font_bold, fontSize=16, spaceAfter=10, textColor=colors.black))
    styles.add(ParagraphStyle(name='CustomHeading2', parent=styles['h2'], fontName=times_font_bold, fontSize=14, spaceAfter=8, textColor=colors.black))
    styles.add(ParagraphStyle(name='CustomNormal', parent=styles['Normal'], fontName=times_font, fontSize=10, spaceAfter=6, textColor=colors.black))
    styles.add(ParagraphStyle(name='CustomSmall', parent=styles['Normal'], fontName=times_font, fontSize=8, spaceAfter=4, textColor=colors.black))

    story = []

    # Title Section
    story.append(Paragraph("Laporan Analisis Pola Klaim Peserta", styles['CustomTitleStyle']))
    story.append(Paragraph("PT ASABRI (Persero) Kancab Batam", styles['CustomSubtitleStyle']))
    story.append(Paragraph("Implementasi K-Means Clustering untuk Efisiensi Pengelolaan Klaim dan Pengambilan Keputusan Berbasis Data", styles['CustomNormal']))
    story.append(Paragraph(f"ID Klasterisasi: {historical_run.id}", styles['CustomSmall']))
    story.append(Paragraph(f"Tanggal Laporan: {historical_run.timestamp.strftime('%d %B %Y %H:%M:%S')}", styles['CustomSmall']))
    story.append(Spacer(1, 0.2 * inch))

    # Introduction
    story.append(Paragraph("1. Pendahuluan", styles['CustomHeading1']))
    story.append(Paragraph("Penelitian ini bertujuan untuk mengimplementasikan sistem identifikasi pola klaim peserta PT ASABRI (Persero) Kancab Batam menggunakan algoritma K-Means Clustering, sebagai upaya untuk meningkatkan efisiensi pengelolaan klaim dan mendukung pengambilan keputusan berbasis data.", styles['CustomNormal']))
    story.append(Paragraph("Identifikasi pola klaim sangat penting untuk memahami karakteristik pensiunan dan jenis klaim yang mereka ajukan. Dengan mengelompokkan pensiunan berdasarkan fitur-fitur relevan, PT ASABRI dapat merancang strategi pengelolaan klaim yang lebih efisien, mengidentifikasi potensi risiko atau kebutuhan khusus, serta membuat keputusan bisnis yang lebih tepat berdasarkan data.", styles['CustomNormal']))
    story.append(Spacer(1, 0.2 * inch))

    # Methodology
    story.append(Paragraph("2. Metodologi Klasterisasi K-Means", styles['CustomHeading1']))
    story.append(Paragraph("Algoritma K-Means adalah metode klasterisasi non-hierarkis yang bertujuan untuk mempartisi n observasi ke dalam k klaster, di mana setiap observasi termasuk dalam klaster dengan rata-rata terdekat (pusat klaster atau centroid). Proses ini melibatkan iterasi untuk meminimalkan jumlah kuadrat jarak antara titik data dan centroid klaster mereka.", styles['CustomNormal']))
    story.append(Paragraph("Fitur-fitur yang digunakan dalam proses klasterisasi ini meliputi:", styles['CustomNormal']))

    features_list_data = []
    for feature_name in historical_run.features_used.split(','):
        feature_type = 'Numerik' if feature_name in ['usia', 'bulan'] else 'Kategorikal'
        features_list_data.append([Paragraph(f"• <b>{feature_name.replace('_', ' ').title()}</b>: {feature_type}.", styles['CustomNormal'])])
    if features_list_data:
        features_table = Table(features_list_data, colWidths=[6.5*inch])
        features_table.setStyle(TableStyle([
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(features_table)
    story.append(Paragraph(f"Jumlah klaster (K) yang dipilih untuk analisis ini adalah: <b>{historical_run.k_value}</b>.", styles['CustomNormal']))
    story.append(Spacer(1, 0.2 * inch))

    # Results and Analysis
    story.append(Paragraph("3. Hasil dan Analisis Klasterisasi", styles['CustomHeading1']))
    story.append(Paragraph("3.1. Kualitas Klaster (Silhouette Score)", styles['CustomHeading2']))
    story.append(Paragraph("Silhouette Score adalah metrik yang digunakan untuk mengevaluasi kualitas klasterisasi. Nilai Silhouette Score berkisar dari -1 hingga 1, di mana nilai yang lebih tinggi menunjukkan klaster yang lebih baik (lebih padat dan terpisah dengan baik).", styles['CustomNormal']))
    story.append(Paragraph(f"Hasil Silhouette Score untuk klasterisasi ini adalah: <b>{historical_run.silhouette_score_value:.2f}</b>.", styles['CustomNormal']))
    story.append(Paragraph("Nilai ini mengindikasikan bahwa klaster-klaster yang terbentuk cukup kohesif dan terpisah satu sama lain.", styles['CustomNormal']))
    story.append(Spacer(1, 0.2 * inch))

    # Visualisasi Klaster
    story.append(Paragraph("3.2. Visualisasi Klaster", styles['CustomHeading2']))
    story.append(Paragraph(f"Visualisasi di bawah ini menampilkan hasil klasterisasi dari run ID {historical_run.id}. Plot ini disajikan dalam bentuk Box Plot atau Scatter Plot, diwarnai berdasarkan klaster.", styles['CustomNormal']))
    if plot_data_json:
        try:
            fig = go.Figure(json.loads(plot_data_json))
            img_buffer = BytesIO()
            fig.write_image(img_buffer, format='png', width=800, height=500, scale=1)
            img_buffer.seek(0)
            img = Image(img_buffer)
            img.drawWidth = 6 * inch
            img.drawHeight = 4 * inch
            story.append(img)
            story.append(Paragraph("Interpretasi Visual: Plot ini membantu memahami bagaimana fitur-fitur pensiunan bervariasi di antara klaster. Perbedaan dalam distribusi antar klaster menunjukkan karakteristik demografi yang berbeda dalam setiap kelompok yang teridentifikasi.", styles['CustomSmall']))
        except Exception as e:
            story.append(Paragraph(f"Gagal memuat visualisasi klaster: {e}", styles['CustomNormal']))
    else:
        story.append(Paragraph("Visualisasi klaster untuk run ini tidak tersedia atau tidak ada variasi data yang cukup untuk plot.", styles['CustomNormal']))
    story.append(Spacer(1, 0.2 * inch))

    # Cluster Profiles
    story.append(Paragraph("3.3. Profil Setiap Klaster", styles['CustomHeading2']))
    if cluster_details:
        for cluster in cluster_details:
            story.append(Paragraph(f"<b>{cluster['name']}</b> (Total Data: {cluster['total_data']})", styles['CustomNormal']))
            story.append(Paragraph(f"Rata-rata Usia: {cluster['rata_rata_usia']:.2f}" if isinstance(cluster['rata_rata_usia'], (int, float)) else f"Rata-rata Usia: {cluster['rata_rata_usia']}", styles['CustomNormal']))
            story.append(Paragraph(f"Rata-rata Bulan: {cluster.get('rata_rata_bulan', 'N/A'):.2f}" if isinstance(cluster.get('rata_rata_bulan', 'N/A'), (int, float)) else f"Rata-rata Bulan: {cluster.get('rata_rata_bulan', 'N/A')}", styles['CustomNormal']))
            story.append(Paragraph("Distribusi Fitur Kategorikal:", styles['CustomNormal']))

            cat_features_table_data = []
            for cat_feature_key, distribution in cluster.items():
                if cat_feature_key.startswith('distribusi_') and distribution:
                    feature_name = cat_feature_key.replace('distribusi_', '').replace('_', ' ').title()
                    cat_features_table_data.append([Paragraph(f"<b>{feature_name}:</b>", styles['CustomNormal'])])
                    for item, count in distribution.items():
                        percentage = (count / cluster['total_data'] * 100) if cluster['total_data'] > 0 else 0
                        cat_features_table_data.append([Paragraph(f"• {item.title()}: {count} data ({percentage:.2f}%)", styles['CustomNormal'])])
            if cat_features_table_data:
                cat_table = Table(cat_features_table_data, colWidths=[6.5*inch])
                cat_table.setStyle(TableStyle([
                    ('LEFTPADDING', (0,0), (-1,-1), 0),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ]))
                story.append(cat_table)
            # PATCH: narasi tanpa label "Narasi Otomatis"
            narasi, rekomendasi = narasi_klaster(cluster)
            story.append(Paragraph(narasi, styles['CustomSmall']))
            if rekomendasi:
                for rek in rekomendasi:
                    story.append(Paragraph(f"Rekomendasi: {rek}", styles['CustomSmall']))
            story.append(Spacer(1, 0.1 * inch))
    else:
        story.append(Paragraph("Detail setiap klaster untuk run ini tidak ditemukan.", styles['CustomNormal']))
    story.append(Spacer(1, 0.2 * inch))

    # Conclusion and Recommendations
    story.append(Paragraph("4. Kesimpulan dan Rekomendasi", styles['CustomHeading1']))
    story.append(Paragraph("Berdasarkan hasil klasterisasi ini, kami dapat mengidentifikasi beberapa pola klaim peserta PT ASABRI (Persero) Kancab Batam. Pola-pola ini dapat dimanfaatkan untuk:", styles['CustomNormal']))
    story.append(Paragraph("• <b>Peningkatan Efisiensi Pengelolaan Klaim:</b> Dengan memahami karakteristik setiap kelompok pensiunan, PT ASABRI dapat mengoptimalkan proses verifikasi, alokasi sumber daya, dan penyelesaian klaim.", styles['CustomNormal']))
    story.append(Paragraph("• <b>Dukungan Pengambilan Keputusan Berbasis Data:</b> Informasi klasterisasi memberikan wawasan yang mendalam untuk pengembangan strategi layanan, identifikasi segmen pensiunan berisiko tinggi atau membutuhkan perhatian khusus, serta perumusan kebijakan yang lebih tepat sasaran.", styles['CustomNormal']))
    story.append(Paragraph("<b>Rekomendasi Spesifik:</b>", styles['CustomNormal']))
    # PATCH: rekomendasi dinamis per klaster
    for cluster in cluster_details:
        _, rekomendasi = narasi_klaster(cluster)
        for rek in rekomendasi:
            story.append(Paragraph(f"• {rek}", styles['CustomNormal']))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("--- INI LIVE CODING ---", styles['CustomSmall']))

    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route('/download_pdf_global_report/<int:cluster_run_id>')
@login_required
def download_pdf_global_report(cluster_run_id):
    pdf_buffer = generate_global_pdf(cluster_run_id)
    if pdf_buffer:
        return send_file(pdf_buffer, as_attachment=True, download_name=f'laporan_klasterisasi_global_run_{cluster_run_id}.pdf', mimetype='application/pdf')
    else:
        flash("Gagal membuat laporan PDF global.", 'danger')
        return redirect(url_for('view_historical_report', cluster_run_id=cluster_run_id))

@app.route('/download_pdf_detail_data/<string:nomor_pensiun>')
@login_required
def download_pdf_detail_data(nomor_pensiun):
    pdf_buffer = generate_detail_pdf(nomor_pensiun)
    if pdf_buffer:
        return send_file(pdf_buffer, as_attachment=True, download_name=f'laporan_detail_pensiun_{nomor_pensiun}.pdf', mimetype='application/pdf')
    else:
        flash("Gagal membuat laporan PDF detail.", 'danger')
        return redirect(url_for('report_detail_page', nomor_pensiun=nomor_pensiun))

def create_tables(): 
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            hashed_password = generate_password_hash('admin', method='scrypt')
            admin_user = User(username='admin', password=hashed_password, nama_lengkap='Administrator', email='admin@asabri.com', role='Admin')
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created: username='admin', password='admin'")
        if not User.query.filter_by(username='kancab').first():
            hashed_password = generate_password_hash('kancab', method='scrypt')
            kancab_user = User(username='kancab', password=hashed_password, nama_lengkap='Kepala Cabang', email='kancab@asabri.com', role='Kancab', cabang='Batam')
            db.session.add(kancab_user)
            db.session.commit()
            print("Kancab user created: username='kancab', password='kancab', cabang='Batam'")
        else:
            existing_kancab = User.query.filter_by(username='kancab').first()
            if not existing_kancab.cabang:
                existing_kancab.cabang = 'Batam'
                db.session.commit()
                print("Existing Kancab user updated with cabang='Batam'")
if __name__ == '__main__':
    create_tables() 
    app.run(debug=True)