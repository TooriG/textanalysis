import streamlit as st
import MeCab
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import networkx as nx

def tokenize(text):
    mecab = MeCab.Tagger("-Owakati")
    return mecab.parse(text).split()

def count_words(text):
    words = tokenize(text)
    return Counter(words)

def calculate_tfidf(text):
    words = tokenize(text)
    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    tf_idf = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names()
    return [(feature_names[i], tf_idf[0, i]) for i in range(len(feature_names))]

def create_wordcloud(word_freq):
    from wordcloud import WordCloud
    wc = WordCloud(background_color="white", font_path="./fonts/YuMincho-Regular.ttf", width=800, height=400)
    wc.generate_from_frequencies(word_freq)
    return wc

def create_graph(tfidf):
    co_matrix = tfidf.T * tfidf
    co_matrix.setdiag(0)
    coo = co_matrix.tocoo()
    G = nx.Graph()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        G.add_edge(feature_names[i], feature_names[j], weight=v)
    return G

st.set_page_config(page_title="日本語文学研究用アプリ", page_icon="📚", layout="wide")

st.title("日本語文学研究用アプリ")

# テキストを入力するためのUI
user_input = st.text_area("テキストを入力してください", "", height=400)

# ユーザーがテキストを入力したら、後続の処理を実行する
if user_input:
    # 単語の頻度をカウントする
    word_freq = count_words(user_input)

    # 単語の重要度を計算する
    tfidf = calculate_tfidf(user_input)

    # 単語の頻度を可視化する
    st.subheader("単語の頻度")
    st.write(word_freq.most_common())

    # ワードクラウドを作成する
    st.subheader("ワードクラウド")
    wc = create_wordcloud(word_freq)
    st.image(wc.to_array(), use_column_width=True)

    # 単語の重要度を可視化する
    st.subheader("単語の重要度（TF-IDF）")
    tfidf_dict = dict(tfidf)
    st.write(sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True))

    # 単語間の関係を可視化する
    st.subheader("単語間の関係")
    G = create_graph(tfidf)
    pos = nx.spring_layout(G, k=0.5)
    nx.draw_networkx_nodes(G, pos, node_color="#ffb347", node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="Yu Mincho")
    nx.draw_networkx_edges(G, pos, width=1)
    nx.draw_networkx_edge_labels(G, pos, font_size=12, font_family="Yu Mincho")
    plt.axis("off")
    st.pyplot(plt)
