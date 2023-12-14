import streamlit as st
import pandas as pd
import string
import nltk
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Import stopwords dalam bahasa Indonesia
from nltk.corpus import stopwords
stop_words = set(stopwords.words('indonesian'))
# Stemmer definition
stemmer = PorterStemmer()
# Stemming
Fact = StemmerFactory()
Stemmer = Fact.create_stemmer()


def correctSlangWords(text, slang_mapping):
    corrected_words = [slang_mapping.get(word, word) for word in text]
    return corrected_words

def load_slang_mapping(file_path):
    slang_mapping = {}
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                key, value = line.split(maxsplit=1)
                slang_mapping[key] = value
            except ValueError:
                print(f"Warning: Invalid format on line {line_number}")
    return slang_mapping
slang_mapping = load_slang_mapping('kbba.txt')

def preprocess_data(df, slang_mapping):
    # Menambahkan kolom untuk setiap tahap preprocessing
    data_cleaned = pd.DataFrame(df['full_text'], columns=['full_text'])

    # Kolom untuk setiap tahap preprocessing
    data_cleaned['removed_handles'] = data_cleaned['full_text'].apply(lambda x: re.sub(r'@[\w]*', '', x))
    data_cleaned['removed_hashtags'] = data_cleaned['removed_handles'].apply(lambda x: re.sub(r'#\w+', '', x))
    data_cleaned['removed_urls'] = data_cleaned['removed_hashtags'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
    data_cleaned['removed_punctuation'] = data_cleaned['removed_urls'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    data_cleaned['lowercase'] = data_cleaned['removed_punctuation'].apply(lambda x: x.lower())
    data_cleaned['removed_emoji'] = data_cleaned['lowercase'].apply(lambda x: re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]+', '', x))
    data_cleaned['tokenized'] = data_cleaned['removed_emoji'].apply(lambda x: nltk.word_tokenize(x))
    data_cleaned['corrected_slang'] = data_cleaned['tokenized'].apply(lambda x: correctSlangWords(x, slang_mapping))
    data_cleaned['removed_stopwords'] = data_cleaned['corrected_slang'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])
    data_cleaned['stemmed'] = data_cleaned['removed_stopwords'].apply(lambda tokens: [Stemmer.stem(word) for word in tokens])
    data_cleaned['removed_numeric'] = data_cleaned['stemmed'].apply(lambda words: [word for word in words if not re.match(r'.*\d.*', word)])
    data_cleaned['cleaned_tweet'] = data_cleaned['removed_numeric'].apply(lambda tokens: ' '.join(tokens))
    return data_cleaned

def load_model(file_path):
    with open(file_path, 'rb') as model_file:
        loaded_model, loaded_vectorizer = pickle.load(model_file)
    return loaded_model, loaded_vectorizer

def label_data(text, model, vectorizer):
    # Transformasi teks menggunakan vectorizer yang telah di-fit
    features = vectorizer.transform([text])
    prediction = model.predict(features)
    return prediction[0]

def main():
    st.title("Aplikasi Tweet Sentiment Analysis")

    # Pilihan untuk menggunakan contoh CSV atau mengunggah CSV sendiri
    upload_option = st.radio("Pilih opsi:", ("Gunakan contoh CSV", "Unggah CSV sendiri"))

    if upload_option == "Gunakan contoh CSV":
        # Gunakan contoh CSV yang sudah disediakan
        df = pd.read_csv('try_tweet.csv', encoding='latin1')
    else:
        # Mengunggah file CSV dari user
        uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

        if uploaded_file is not None:
            # Membaca file CSV menjadi DataFrame
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            st.warning("Silakan unggah file CSV atau pilih opsi untuk menggunakan contoh CSV. Menggunakan file unggahan memakan waktu lebih lama")
            # Menghentikan eksekusi selanjutnya jika tidak ada file yang diunggah
            return

    # Menampilkan data DataFrame
    st.subheader("Data yang diimpor:")
    st.dataframe(df)

    # Tombol lanjutan
    if st.button("Lanjut ke proses selanjutnya"):
        # Menghapus duplikat berdasarkan kolom 'full_text'
        df_no_duplicates = df.drop_duplicates(subset='full_text').copy()

        # Preprocessing data
        st.sidebar.text("Preprocessing data...")
        df_preprocessed = preprocess_data(df_no_duplicates, slang_mapping)
        st.subheader("Data setelah preprocessing:")
        st.dataframe(df_preprocessed)

        # Memuat model menggunakan pickle
        model_path = 'svm_model.pkl'  # Ganti dengan path dan nama file model Anda
        loaded_model, loaded_vectorizer = load_model(model_path)

        # Pelabelan otomatis
        st.sidebar.text("Melakukan pelabelan otomatis...")
        df_preprocessed['predicted_label'] = df_preprocessed['cleaned_tweet'].apply(lambda x: label_data(x, loaded_model, loaded_vectorizer))

        # Menyatukan data awal dan kolom predicted_label
        df_result = pd.concat([df, df_preprocessed['predicted_label']], axis=1)

        # Menampilkan hasil
        st.subheader("Data Awal dengan Label yang Sudah Diprediksi:")
        st.dataframe(df_result[['full_text', 'predicted_label']])

        # Visualisasi pie chart
        st.subheader("Visualisasi Hasil Prediksi Label:")
        labels_count = df_result['predicted_label'].value_counts()

        plt.figure(figsize=(10, 6))
        plt.pie(labels_count, labels=labels_count.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis'))
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Persentase Prediksi Label')
        st.pyplot(plt)

        # Save labeled data to CSV
        st.sidebar.text("Menyimpan data hasil pelabelan...")
        df_preprocessed.to_csv('labeled_data.csv', index=False)
        st.sidebar.text("Proses selesai!")

if __name__ == "__main__":
    main()

