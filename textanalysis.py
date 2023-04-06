import streamlit as st
import MeCab
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx

# MeCabの初期化
mecab = MeCab.Tagger("-Owakati")

def extract_nouns(text):
    """
    文章から名詞を抽出する関数
    """
    node = mecab.parse(text)
    words = node.split(" ")
    nouns = [word for word in words if "名詞" in mecab.parse(word).split("\t")[1]]
    return nouns

def main():
    st.title("日本語テキスト解析アプリ")

    # テキストの入力
    text = st.text_area("解析したいテキストを入力してください。", height=200)

    # 名詞の抽出
    nouns = extract_nouns(text)

    # 名詞の出現頻度の分析
    st.header("名詞の出現頻度")
    st.write("総単語数:", len(nouns))
    freq_df = pd.DataFrame({"単語":nouns})
    freq_df = freq_df.groupby("単語").size().reset_index(name="出現回数")
    freq_df = freq_df.sort_values("出現回数", ascending=False)
    st.dataframe(freq_df)

    # WordCloudの作成
    st.header("WordCloud")
    wc = WordCloud(width=800, height=400, background_color="white", font_path="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf").generate(" ".join(nouns))
    plt.figure(figsize=(12, 12))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # LDAによるトピックモデリング
    st.header("トピックモデリング")
    n_topics = st.slider("トピック数を選択してください。", 1, 10, 3)
    vectorizer = CountVectorizer(tokenizer=extract_nouns)
    x = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=n_topics)
    lda.fit(x)
    feature_names = vectorizer.get_feature_names()
    for i, topic in enumerate(lda.components_):
        st.subheader(f"トピック{i+1}")
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        st.write(", ".join(top_words))

        # 単語共起ネットワーク
    st.header("単語共起ネットワーク")
    window_size = st.slider("窓サイズを選択してください。", 1, 10, 2)
    words = [word for word in text.split() if word in nouns]
    cooccurrence = np.zeros((len(set(nouns)), len(set(nouns))))
    for i in range(len(words)):
        for j in range(i+1, min(i+window_size+1, len(words))):
            if words[j] != words[i]:
                cooccurrence[list(set(nouns)).index(words[i]), list(set(nouns)).index(words[j])] += 1
                cooccurrence[list(set(nouns)).index(words[j]), list(set(nouns)).index(words[i])] += 1
    G = nx.Graph()
    for i in range(len(nouns)):
        G.add_node(nouns[i])
    for i in range(len(nouns)):
        for j in range(i+1, len(nouns)):
            if cooccurrence[i, j] > 0:
                G.add_edge(nouns[i], nouns[j], weight=cooccurrence[i, j])
    pos = nx.spring_layout(G, k=0.7)
    nx.draw_networkx_nodes(G, pos, node_size=300)
    nx.draw_networkx_labels(G, pos, font_family="IPAexGothic")
    edges = [(u, v) for (u, v, d) in G.edges(data=True)]
    weights = [d["weight"] for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.5, edge_color="r")
    plt.axis("off")
    st.pyplot(plt)
