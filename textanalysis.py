import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import MeCab

st.title("日本語文学研究用の分析アプリ")

# ユーザーが入力したテキストを取得
user_input = st.text_area("テキストを入力してください", "")

if user_input:
    # テキストを前処理
    mecab = MeCab.Tagger("-Owakati")
    preprocessed_text = mecab.parse(user_input)

    # 単語の頻度を計算
    words = preprocessed_text.split()
    word_counts = dict(Counter(words))
    word_count_df = pd.DataFrame(
        list(word_counts.items()),
        columns=["単語", "出現回数"]
    ).sort_values("出現回数", ascending=False)

    # 単語の共起関係を計算
    cooc_matrix = CountVectorizer().fit_transform([preprocessed_text]).toarray()
    vocab = CountVectorizer().fit([preprocessed_text]).get_feature_names()
    cooc_df = pd.DataFrame(
        cooc_matrix,
        columns=vocab,
        index=vocab
    )
    cooc_df = cooc_df.apply(lambda x: np.log(x+1))

    # 共起グラフを作成
    cooc_graph = nx.from_numpy_array(cooc_df.values)

    # 可視化のための設定
    pos = nx.spring_layout(cooc_graph, k=0.2, seed=1)

    # 結果を表示
    st.write("単語の出現回数")
    st.write(word_count_df)

    st.write("単語の共起関係")
    st.write(cooc_df)

    st.write("単語の共起グラフ")
    plt.figure(figsize=(10,10))
    nx.draw_networkx_nodes(cooc_graph, pos, node_color="r", alpha=0.8, node_size=100)
    nx.draw_networkx_edges(cooc_graph, pos, edge_color="gray", alpha=0.5, width=cooc_df.values.flatten()*0.5)
    nx.draw_networkx_labels(cooc_graph, pos, font_family="IPAexGothic")
    plt.axis("off")
    st.pyplot()
