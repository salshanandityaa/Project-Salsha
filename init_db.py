import sqlite3
import os # Import module os untuk memeriksa keberadaan file

DATABASE_NAME = 'database.db' # Ganti nama database agar sesuai dengan app.py Anda, jika berbeda

def init_db():
    # Pastikan file database.db dihapus jika ada, untuk inisialisasi bersih
    # Ini opsional, hanya jika Anda ingin selalu memulai dari database kosong
    if os.path.exists(DATABASE_NAME):
        os.remove(DATABASE_NAME)
        print(f"File database '{DATABASE_NAME}' yang lama dihapus.")

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    print(f"Membuat tabel di database '{DATABASE_NAME}'...")

    # 1. Membuat tabel 'users'
    # Sesuai ERD, password di ERD adalah password_hash, dan role default 'user'
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL, -- Ganti dari 'password' menjadi 'password_hash'
        role TEXT DEFAULT 'user' NOT NULL -- Tambahkan DEFAULT 'user' sesuai ERD
    )
    ''')
    print("Tabel 'users' berhasil dibuat/diverifikasi.")

    # 2. Membuat tabel 'cluster'
    # Pastikan nama kolom 'label_cluster' cocok dengan ERD
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cluster (
        id_cluster INTEGER PRIMARY KEY, -- ERD menunjukkan id_cluster sebagai PK dan INTEGER
        label_cluster TEXT NOT NULL
    )
    ''')
    print("Tabel 'cluster' berhasil dibuat/diverifikasi.")

    # 3. Membuat tabel 'jenis_pensiun'
    # Pastikan nama kolom 'nama_jenis_pensiun' cocok dengan ERD
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS jenis_pensiun (
        id_jenis_pensiun INTEGER PRIMARY KEY AUTOINCREMENT,
        nama_jenis_pensiun TEXT NOT NULL UNIQUE
    )
    ''')
    print("Tabel 'jenis_pensiun' berhasil dibuat/diverifikasi.")

    # 4. Membuat tabel 'data_belum_auten'
    # Perhatikan semua kolom dan FOREIGN KEY sesuai ERD
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS data_belum_auten (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nomor_pensiun TEXT,
        penerima TEXT,
        status_pensiun_text TEXT,
        cabang TEXT,
        mitra TEXT,
        status_auten REAL,
        waktu REAL,
        bulan REAL,
        jenis_pekerjaan_text TEXT,
        usia REAL,
        jenis_pekerjaan_encoded INTEGER,
        status_pensiun_encoded INTEGER,
        status_auten_encoded INTEGER,
        cluster INTEGER DEFAULT -1, -- Nilai default -1 jika belum di-cluster
        id_jenis_pensiun INTEGER,
        user_id INTEGER, -- Sesuaikan dengan nama kolom di tabel users (jika id, maka pakai id)
        FOREIGN KEY (id_jenis_pensiun) REFERENCES jenis_pensiun(id_jenis_pensiun),
        FOREIGN KEY (user_id) REFERENCES users(id) -- Merujuk ke kolom 'id' di tabel 'users'
    )
    ''')
    print("Tabel 'data_belum_auten' berhasil dibuat/diverifikasi.")

    # 5. Membuat tabel 'data_auten'
    # Perhatikan semua kolom dan FOREIGN KEY sesuai ERD
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS data_auten (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nomor_pensiun TEXT,
        penerima TEXT,
        status_pensiun_text TEXT,
        cabang TEXT,
        mitra TEXT,
        status_auten REAL,
        waktu REAL,
        bulan REAL,
        jenis_pekerjaan_text TEXT,
        usia REAL,
        jenis_pekerjaan_encoded INTEGER,
        status_pensiun_encoded INTEGER,
        status_auten_encoded INTEGER,
        cluster INTEGER DEFAULT -1, -- Nilai default -1 jika belum di-cluster
        id_jenis_pensiun INTEGER,
        user_id INTEGER,
        FOREIGN KEY (id_jenis_pensiun) REFERENCES jenis_pensiun(id_jenis_pensiun),
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    print("Tabel 'data_auten' berhasil dibuat/diverifikasi.")

    # 6. Membuat tabel 'laporan_analisis'
    # Pastikan penamaan kolom dan tipe data sesuai ERD
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS laporan_analisis (
        id_laporan INTEGER PRIMARY KEY AUTOINCREMENT,
        tanggal DATE,
        id_cluster INTEGER,
        id_jenis_pensiun INTEGER,
        FOREIGN KEY (id_cluster) REFERENCES cluster(id_cluster),
        FOREIGN KEY (id_jenis_pensiun) REFERENCES jenis_pensiun(id_jenis_pensiun)
    )
    ''')
    print("Tabel 'laporan_analisis' berhasil dibuat/diverifikasi.")

    # Opsional: Sisipkan data awal untuk 'jenis_pensiun' dan 'cluster'
    # Ini penting agar foreign key tidak kosong saat pengujian
    cursor.execute("INSERT OR IGNORE INTO jenis_pensiun (id_jenis_pensiun, nama_jenis_pensiun) VALUES (1, 'Normal')")
    cursor.execute("INSERT OR IGNORE INTO jenis_pensiun (id_jenis_pensiun, nama_jenis_pensiun) VALUES (2, 'Dipercepat')")
    cursor.execute("INSERT OR IGNORE INTO jenis_pensiun (id_jenis_pensiun, nama_jenis_pensiun) VALUES (3, 'Cacat')")
    cursor.execute("INSERT OR IGNORE INTO jenis_pensiun (id_jenis_pensiun, nama_jenis_pensiun) VALUES (4, 'Janda/Duda')")
    cursor.execute("INSERT OR IGNORE INTO jenis_pensiun (id_jenis_pensiun, nama_jenis_pensiun) VALUES (5, 'Tunda')")
    cursor.execute("INSERT OR IGNORE INTO jenis_pensiun (id_jenis_pensiun, nama_jenis_pensiun) VALUES (6, 'Tidak diketahui')")
    print("Data awal 'jenis_pensiun' disisipkan/diverifikasi.")

    # Contoh label cluster (Anda bisa sesuaikan nanti setelah clustering)
    cursor.execute("INSERT OR IGNORE INTO cluster (id_cluster, label_cluster) VALUES (0, 'Cluster 0')")
    cursor.execute("INSERT OR IGNORE INTO cluster (id_cluster, label_cluster) VALUES (1, 'Cluster 1')")
    cursor.execute("INSERT OR IGNORE INTO cluster (id_cluster, label_cluster) VALUES (2, 'Cluster 2')")
    print("Data awal 'cluster' disisipkan/diverifikasi.")


    conn.commit()
    conn.close()
    print("Semua tabel dan data awal berhasil dibuat di database.")

if __name__ == '__main__':
    init_db()