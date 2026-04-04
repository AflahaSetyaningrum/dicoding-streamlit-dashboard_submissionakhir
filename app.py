@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)

    possible_files = [
        os.path.join(base_path, "data", "students.csv"),
        os.path.join(base_path, "data", "students_performance.csv"),
        os.path.join(base_path, "students.csv"),
        os.path.join(base_path, "students_performance.csv"),
    ]

    for file in possible_files:
        if os.path.exists(file):

            # coba baca dengan delimiter ;
            df = pd.read_csv(file, delimiter=";")

            # kalau cuma 1 kolom → berarti salah parsing
            if df.shape[1] == 1:
                df = pd.read_csv(file)  # fallback ke koma

            return df

    st.error("❌ Dataset tidak ditemukan")
    st.stop()