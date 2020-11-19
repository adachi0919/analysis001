import requests
from bs4 import BeautifulSoup
import re
import pandas as pd, numpy as np
import unicodedata
import math
import string
from bs4 import BeautifulSoup
import requests
import jaconv
from gensim import corpora
from gensim import models
from pprint import pprint
import MeCab
from gensim.models.doc2vec import Doc2Vec, TaggedDocument



# MeCabの辞書にNEologdを指定。
# mecabは携帯素解析用、wakatiは分かち書き用
mecab = MeCab.Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/')
wakati = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/")

# 形態素解析を行う関数を定義
# ファイルを入力するとファイルを出力し、文字列を渡すと文字列を返します。引数fileで変更します。
# 単に分かち書きしたいだけの場合は引数にmecab=wakatiとすると実現できます。
def MecabMorphologicalAnalysis(path='./text.txt', output_file='wakati.txt', mecab=mecab, file=False):
    mecab_text = ''
    if file:
        with open(path) as f:
            for line in f:
                mecab_text += mecab.parse(line)
        with open(output_file, 'w') as f:
            print(mecab_text, file=f)
    else:
        for path in path.split('\n'):
            mecab_text += mecab.parse(path)
        return mecab_text



# 記号文字は分析をするにあたって邪魔になるため、記号を取り除く関数を定義します。
def symbol_removal(soup):
    soup = unicodedata.normalize("NFKC", soup)
    exclusion = "「」『』【】〈〉《》≪≫、。・◇◆■●" + "\n" + "\r" + "\u3000" # 除去する記号文字を指定
    soup = soup.translate(str.maketrans("", "", string.punctuation  + exclusion))
    return soup



# 青空文庫の情報をスクレイピングして、テーブルデータに整形する処理を行う関数を定義します。  
# 引数に指定した数のタイトルを出力します。(デフォルトは30)  
# 中でsymbol_removal関数を使用しています。
def Aozora_table(n=30):
    url = "https://www.aozora.gr.jp/access_ranking/2019_xhtml.html"
    res = requests.get(url)
    res.encoding = 'shift-jis'
    soup = BeautifulSoup(res.content, "html.parser")

    url_list = [url["href"] for i, url in enumerate(soup.find_all("a", target="_blank")) if i < n]

    title = []
    category = []
    text = []
    for url in url_list:
        res = requests.get(url)
        url_start = url[:37]
        res.encoding = 'shift-jis'
        soup = BeautifulSoup(res.content, "html.parser")
        for i, a in enumerate(soup.find_all("a")):
            if i == 7:
                url_end = a["href"][1:]
        url = url_start + url_end
        res = requests.get(url)
        res.encoding = 'shift-jis'
        soup = BeautifulSoup(res.content, "html.parser")
        title.append(soup.find("h1").string)
        category.append(soup.find("h2").string)
        for tag in soup.find_all(["rt", "rp"]):
            tag.decompose()
        soup = soup.find("div",{'class': 'main_text'}).get_text()
        text.append(symbol_removal(soup))
    df = pd.DataFrame({'title': title, 'category': category, 'text': text})
    return df


# Yahooニュースをスクレイピングする関数です。
# 引数で指定した数の記事をとってきてデータフレームを返します。
# 中でsymbol_removal関数を使用しています。
def YahooNews(n=30):
    url = "https://news.yahoo.co.jp/topics/top-picks"
    URL = "https://news.yahoo.co.jp/"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    all_page_links = []
    all_page_links.append(url)
    all_links = []
    while True:
        try:
            next = soup.find("li", class_="pagination_item-next").find("a")["href"]
            next_link = URL + next
            all_page_links.append(next_link)
            next_res = requests.get(next_link)
            soup = BeautifulSoup(next_res.text, "html.parser")
        except:
            break
            
    title_list = []
    category_list = []
    text_list = []
    for url in all_page_links: # all_page_links: 全てのニュースのリスト
            res = requests.get(url) # url: 25個分のニュースのリスト
            soup = BeautifulSoup(res.text, "html.parser")
            page_soup = soup.find_all("a", class_="newsFeed_item_link")
            for href in page_soup:
                link = href["href"] # link: 一つのニュースのリンク(本文は一部のみ)
                all_links.append(link)
    
    if len(all_links) <= n:
        n = len(all_links)
    
    i = 0
    for link in all_links:
        link_res = requests.get(link)
        href_soup = BeautifulSoup(link_res.text, "html.parser")
        try:
            title = href_soup.find("h1", class_=re.compile("^sc")).string
        except:
            continue
        title_link = href_soup.find("a", class_="sc-fUKxqW")["href"] # title_link: 本文
        res = requests.get(title_link)
        soup = BeautifulSoup(res.text, "html.parser")

        category = soup.find_all("li", class_="current")
        try:
            category = category[1].string
        except:
            continue
        else:
            for tag in soup.find_all(["a"]):
                tag.decompose()
            try:
                soup = soup.find("div", class_="article_body").get_text()
                soup = symbol_removal(soup)
                
                text_list.append(soup)
                title_list.append(title)
                category_list.append(category)
                i += 1 # 本文が正常に保存できたことをトリガーにしてカウントを一つ増やすことにします。
                pro_bar = ('=' * math.ceil(i / (n / 20))) + (' ' * int((n / (n / 20)) - math.ceil(i / (n / 20))))
                print('\r[{0}] {1}記事'.format(pro_bar, i), end='')
                if i >= n:
                    df = pd.DataFrame({'title': title_list, 'category': category_list, 'text': text_list})
                    return df
            except:
                continue
    df = pd.DataFrame({'title': title_list, 'category': category_list, 'text': text_list})
    return df



# 分かち書きされた2階層の単語のリストを渡すことで、TF-IDFでソートされたユニークな単語のリストを得る。
def sortedTFIDF(sentences):
    
    # 単語にIDを添付します。
    dictionary = corpora.Dictionary(sentences)
    
    # 作品ごとの単語の出現回数をカウント
    corpus = list(map(dictionary.doc2bow, sentences))
    
    # 単語ごとにTF-IDFを算出
    test_model = models.TfidfModel(corpus)
    corpus_tfidf = test_model[corpus]
    
    # ID:TF-IDF → TF-IDF:単語 に変換。TF-IDFを左に持ってくることで、sortedを用いてTF-IDFを基準にソートすることができます。
    texts_tfidf = []
    for doc in corpus_tfidf:
        text_tfidf = []
        for word in doc:
            text_tfidf.append([word[1], dictionary[word[0]]])
        texts_tfidf.append(text_tfidf)
    
    # TF-IDFを基準にソートを行います。
    sorted_texts_tfidf = []
    for text in texts_tfidf:
        sorted_text = sorted(text, reverse=True)
        sorted_texts_tfidf.append(sorted_text)

    return sorted_texts_tfidf



# v1とv2のコサイン類似度を出力します。
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))