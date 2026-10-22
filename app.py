
import streamlit as st
import pandas as pd
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocess

st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="💬",
    layout="wide"
)

st.title("Analisis Sentimen Cuitan X")
st.write(
    "Klasifikasi cuitan ke dalam kategori "
    "Hate Speech atau Non Hate Speech."
)

@st.cache_resource
def load_model():
    with open("vokabuler.pkl", "rb") as file:
        vocabulary = pickle.load(file)

    model = joblib.load("DT.sav")
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)

    return model, vectorizer

model, vectorizer = load_model()

tab1, tab2 = st.tabs([
    "Prediksi Cuitan",
    "Prediksi File CSV"
])

with tab1:
    tweet = st.text_area("Masukkan cuitan")

    with st.expander("📝 Lihat Contoh Cuitan"):
        st.markdown("**1. Hate Speech**")
        st.write(
            "@gemanawacita @MUIPusat @mohmahfudmd "
            "@YaqutCQoumas saya sangat setuju MUI itu dibubarkan "
            "biar tidak menjadi tempat berlindungnya HTI dan "
            "Teroris serta Radikalis"
        )

        if st.button("Gunakan Contoh Hate Speech"):
            st.session_state["contoh_tweet"] = (
                "@gemanawacita @MUIPusat @mohmahfudmd "
                "@YaqutCQoumas saya sangat setuju MUI itu dibubarkan "
                "biar tidak menjadi tempat berlindungnya HTI dan "
                "Teroris serta Radikalis"
            )
            st.rerun()

        st.markdown("**2. Non Hate Speech**")
        st.write(
            "@CNNIndonesia Teroris KKB ga ada rencana "
            "kemah di Kalteng nih???"
        )

        if st.button("Gunakan Contoh Non Hate Speech"):
            st.session_state["contoh_tweet"] = (
                "@CNNIndonesia Teroris KKB ga ada rencana "
                "kemah di Kalteng nih???"
            )
            st.rerun()

    if "contoh_tweet" in st.session_state:
        tweet = st.text_area(
            "Cuitan yang dipilih",
            value=st.session_state["contoh_tweet"],
            key="tweet_pilihan"
        )

    if st.button("Analisis Sentimen"):
        if tweet.strip():
            cleaned = preprocess(tweet)

            # Sesuaikan dengan preprocessing saat training.
            features = vectorizer.fit_transform([cleaned])

            labels = {
                "Non_HS": "Non Hate Speech",
                "HS": "Hate Speech"
            }

            prediction = model.predict(features)[0]
            result = labels[prediction]

            if prediction == "HS":
                st.error(f"{result}")
            else:
                st.success(f"{result}")

        else:
            st.warning("Masukkan cuitan terlebih dahulu.")


with tab2:
    uploaded_file = st.file_uploader(
        "Unggah file CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            if "tweet" not in data.columns:
                st.error("CSV harus memiliki kolom 'tweet'.")
            else:
                st.caption(f"Jumlah data: {len(data):,} cuitan")

                if st.button("Analisis Semua Cuitan"):
                    try:
                        with st.status(
                            "Memproses data cuitan...",
                            expanded=True
                        ) as status:

                            st.write("⏳ Membersihkan dan memproses teks...")
                            data["tweet_bersih"] = (
                                data["tweet"]
                                .fillna("")
                                .astype(str)
                                .apply(preprocess)
                            )

                            st.write("⏳ Melakukan ekstraksi fitur TF-IDF...")
                            features = vectorizer.fit_transform(
                                data["tweet_bersih"]
                            )

                            st.write("⏳ Memprediksi sentimen...")
                            predictions = model.predict(features)

                            labels = {
                                "Non_HS": "Non Hate Speech",
                                "HS": "Hate Speech"
                            }

                            data["sentimen"] = [
                                labels[p] for p in predictions
                            ]

                            status.update(
                                label="Analisis sentimen selesai!",
                                state="complete",
                                expanded=False
                            )

                        st.success(
                            f"Berhasil menganalisis {len(data):,} cuitan."
                        )

                        st.dataframe(
                            data.drop(columns=["tweet_bersih"]),
                            use_container_width=True
                        )

                        st.download_button(
                            "Unduh Hasil CSV",
                            data.to_csv(index=False).encode("utf-8"),
                            file_name="hasil_sentimen.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"Gagal memproses data: {e}")

        except Exception as e:
            st.error(f"Gagal membaca file CSV: {e}")
