# util.py(복구 버전)
import pandas as pd
import params as PARAMS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import opinion_lexicon
from sklearn.preprocessing import MinMaxScaler
import csv
from transformers import BertTokenizer
#from kobert_tokenizer import KoBERTTokenizer

# Polarity 사전 로딩
POS_LEX = set(opinion_lexicon.positive())
NEG_LEX = set(opinion_lexicon.negative())



def normalize_added_features(df: pd.DataFrame):
    """
    df 안에서 *_polarity, *_mrc_conc, *_mrc_fam, *_local_idf, *_tf_score
    열들만 골라서 MinMax 정규화(0~1).
    """
    scaler = MinMaxScaler()

    # 정규화 대상 컬럼만 찾기
    target_suffixes = ["_polarity", "_mrc_conc", "_mrc_fam", "_local_idf", "_tf_score"]
    columns_to_scale = [
        col for col in df.columns
        if any(col.endswith(suf) for suf in target_suffixes)
    ]

    if not columns_to_scale:
        print("[INFO] No columns found for normalization.")
        return df

    # 실제로 fit_transform 적용
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df


def load_mrc_data(mrc_file="mrc_database.csv"):
    
    """
    mrc_data.csv 예시:
    word,concreteness,familiarity
    apple,620,500
    home,530,700
    ...
    """
    mrc_dict = {}
    with open(mrc_file, "r") as f:      #, encoding="utf-8"
        reader = csv.DictReader(f)
        for row in reader:
            w = row["Word"].lower().strip()
            conc = float(row["Concreteness"])
            fam  = float(row["Familiarity"])
            mrc_dict[w] = {"conc": conc, "fam": fam}

        
    return mrc_dict


MRC_DICT = load_mrc_data()



def analyze_sentence(sentence, idf_scores, mrc_dict):
    """
    sentence: 단일 문장(str)
    idf_scores: {'apple': 5.2, ...} (이미 util.py에서 구함)
    mrc_dict:   {'apple': {'conc':620, 'fam':500}, ...}
    """
    words = sentence.lower().split()
    word_count = len(words)
    
    # 1) Polarity: (pos_count - neg_count)
    pos_count = sum(1 for w in words if w in POS_LEX)
    neg_count = sum(1 for w in words if w in NEG_LEX)
    polarity = pos_count - neg_count

    # 2) MRC (concreteness, familiarity) 평균
    conc_sum = 0.0
    fam_sum  = 0.0
    mrc_hits = 0
    for w in words:
        if w in mrc_dict:
            conc_sum += mrc_dict[w]["conc"]
            fam_sum  += mrc_dict[w]["fam"]
            mrc_hits += 1
    avg_conc = conc_sum / mrc_hits if mrc_hits else 0.0
    avg_fam  = fam_sum  / mrc_hits if mrc_hits else 0.0

    # 3) IDF: 단어별 IDF 합의 평균
    if word_count > 0:
        idf_sum = sum(idf_scores.get(w,1.0) for w in words)
        avg_idf_local = idf_sum / word_count
    else:
        avg_idf_local = 0.0

    # 4) TF: 아주 간단히 "word_count"를 TF로 볼 수도 있고,
    #        혹은 "단어별 빈도 / word_count" 평균을 쓸 수도 있음
    tf_score = float(word_count)  # 일단은 문장 길이를 TF로 취급

    return {
        "polarity": polarity,
        "avg_conc": avg_conc,
        "avg_fam":  avg_fam,
        "local_idf": avg_idf_local,
        "tf_score": tf_score
    }

def read_data():
    def compute_idf(df):
        def preprocess_text(text):
            """Remove numbers and keep only words in a sentence."""
            return ' '.join([word for word in text.split() if not any(char.isdigit() for char in word)])

        corpus = []
        """Compute IDF scores for words across all text fields in the dataset."""
        for index, row in df.iterrows():
            for category in PARAMS.FEATURES:
                if category == "Patient_ID" or PARAMS.FULL_FEATURES[category] == 'int32':
                    continue
                sentences = row[category]
                if isinstance(sentences, list) and sentences:
                    # Preprocess each sentence to remove numbers before adding to corpus
                    cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]
                    corpus.extend(cleaned_sentences)

        vectorizer = TfidfVectorizer(use_idf=True)
        vectorizer.fit(corpus)  # Learn IDF values from the corpus
        idf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
        
        return idf_scores
    

    def compute_max_sentences(df):
        """Compute the maximum number of sentences per category across the entire dataset."""
        max_sentences_per_category = {
            category: df[category].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
            for category in PARAMS.FEATURES
            if category != "Patient_ID" and PARAMS.FULL_FEATURES[category] != 'int32'
        }
        return max_sentences_per_category

    def compute_max_token_length(df):
        #from transformers import DistilBertTokenizer
        #tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        #tokenizer = KoBERTTokenizer.from_pretrained('monologg/kobert')
        tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
        """Compute the maximum token length per category using DistilBERT tokenization."""
        max_token_length_per_category = {
            category: df[category].apply(lambda x: max((len(tokenizer.tokenize(sentence)) for sentence in x), default=0)
            if isinstance(x, list) else 0).max()
            for category in PARAMS.FEATURES
            if category != "Patient_ID" and PARAMS.FULL_FEATURES[category] != 'int32'
        }
        return max_token_length_per_category

    def extract_statistics(df, idf_scores):
        """
        기존 코드 + Polarity/MRC/TF등 추가
        """
        # 먼저 MRC_DICT를 사용하기 위해 전역 변수를 가져오거나, 
        # 필요시 함수 인자로 받은 뒤 MRC_DICT = load_mrc_data() 할 수도 있음

        result = []

        for index, row in df.iterrows():
            row_stats = {}
            for category in PARAMS.FEATURES:
                if category == "Patient_ID" or PARAMS.FULL_FEATURES[category] == 'int32':
                    continue

                sentences = row[category]

                if isinstance(sentences, list) and sentences:
                    total_sentences = len(sentences)
                    total_words = sum(len(sentence.split()) for sentence in sentences)
                    total_chars = sum(len(sentence) for sentence in sentences)
                    total_numbers = sum(len(re.findall(r'\d+', sentence)) for sentence in sentences)

                    avg_words = total_words / total_sentences
                    avg_chars = total_chars / total_sentences

                    # 기존 IDF-weighted sum
                    idf_weighted_sum = sum(
                        sum(idf_scores.get(word, 1) for word in sentence.split()) for sentence in sentences)
                    avg_idf_weight = idf_weighted_sum / total_words if total_words > 0 else 0

                    # ▶ 추가: Polarity/MRC/TF (문장단위)
                    polarity_sum = 0.0
                    conc_sum = 0.0
                    fam_sum  = 0.0
                    idf_loc_sum = 0.0
                    tf_sum = 0.0

                    for sent in sentences:
                        feats = analyze_sentence(sent, idf_scores, MRC_DICT)
                        polarity_sum += feats["polarity"]
                        conc_sum     += feats["avg_conc"]
                        fam_sum      += feats["avg_fam"]
                        idf_loc_sum  += feats["local_idf"]
                        tf_sum       += feats["tf_score"]

                    # 문장별로 계산된 값의 평균
                    avg_polarity = polarity_sum / total_sentences
                    avg_conc     = conc_sum     / total_sentences
                    avg_fam      = fam_sum      / total_sentences
                    avg_idf_local= idf_loc_sum  / total_sentences
                    avg_tf       = tf_sum       / total_sentences

                else:
                    # 문장이 없을 경우 0 처리
                    total_sentences = 0
                    avg_words = 0
                    avg_chars = 0
                    total_numbers = 0
                    avg_idf_weight = 0

                    avg_polarity = 0
                    avg_conc = 0
                    avg_fam  = 0
                    avg_idf_local = 0
                    avg_tf = 0

                # 기존 5개
                row_stats[f'{category}_avg_words'] = avg_words
                row_stats[f'{category}_total_sentences'] = total_sentences
                row_stats[f'{category}_avg_chars'] = avg_chars
                row_stats[f'{category}_total_numbers'] = total_numbers
                row_stats[f'{category}_avg_idf_weight'] = avg_idf_weight

                # 추가 5개
                row_stats[f'{category}_polarity']    = avg_polarity
                row_stats[f'{category}_mrc_conc']    = avg_conc
                row_stats[f'{category}_mrc_fam']     = avg_fam
                row_stats[f'{category}_local_idf']   = avg_idf_local
                row_stats[f'{category}_tf_score']    = avg_tf
                

            result.append(row_stats)

        return pd.DataFrame(result)


    # df = pd.read_csv(PARAMS.DATASET_PATH, encoding_errors="ignore")
    df_ad = pd.read_csv("./data/dataset/AD_250120_korea.csv", encoding_errors="ignore")
    df_mci = pd.read_csv("./data/dataset/MCI_250120_korea.csv", encoding_errors="ignore")
    df_nc = pd.read_csv("./data/dataset/NC_250120_korea.csv", encoding_errors="ignore")
    #df_mix = pd.read_csv("./data/250120/added_format.csv", encoding_errors="ignore")
    #df_nc2 = df_mix[df_mix['Label'] == 'NC']
    #df_nc = pd.concat([df_nc, df_nc2], ignore_index=True)
    #df_mci2 = df_mix[df_mix['Label'] == 'MCI']
    #df_mci = pd.concat([df_mci, df_mci2], ignore_index=True)
    #df_ad2 = df_mix[df_mix['Label'] == 'Dementia']
    #df_ad = pd.concat([df_ad, df_ad2], ignore_index=True)
    df_nc['Label'] = "NC"
    df_mci['Label'] = "MCI"
    df_ad['Label'] = "AD"
    df = pd.concat([df_nc, df_mci, df_ad], ignore_index=True)
    # Convert string representations back to lists
    df = df[["Label", "Original_ID"] + PARAMS.FEATURES]
    for col in PARAMS.FEATURES:
        if col in ["Patient_ID", "Gender", "Age", "Edu"]:
            continue
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else x)
    df = df.applymap(lambda x: [] if isinstance(x, float) and pd.isna(x) else x)
    df = df[
        ~((df['Age'].apply(lambda x: isinstance(x, list) and len(x) == 0)) |
          (df['Edu'].apply(lambda x: isinstance(x, list) and len(x) == 0)))
    ]
    df = df.astype({'Original_ID': 'str', 'Patient_ID': 'Int32', 'Gender': 'Int32', 'Age': 'Int32', 'Edu': 'Int32'})

    # df.nlargest(5, 'Edu')
    df = df[df['Edu'] <= 50]
    df = df[df['Age'] >= 30]
    df = df.dropna(subset=PARAMS.FEATURES)
    df = df.reset_index(drop=True)

    idf_scores = compute_idf(df)  # Compute IDF scores from the dataset
    stats_df = extract_statistics(df, idf_scores)

    # label_encoder = LabelEncoder()
    # df['Gender'] = label_encoder.fit_transform(df['Gender'])

    # df['Age'] = df['Age'].astype("int").astype("str")
    # columns = ["Label"] + PARAMS.FEATURES
    # df = df.astype(PARAMS.FULL_FEATURES)
    # df = df[columns]
    # max_lengths = df.applymap(lambda x: len(str(x))).max()
    df = pd.concat([df, stats_df], axis=1)
    df = normalize_added_features(df)
    df['label_encoded'] = df['Label'].map({cls: i for i, cls in enumerate(PARAMS.CLASSES)})

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)

    return train_df, val_df