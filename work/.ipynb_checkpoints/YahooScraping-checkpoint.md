Yahooニュースをスクレイピングする関数を定義したので、シェアします。  
データフレームを返してくれるので、データ分析に使えます！！  

requests、BeautifulSoupを利用しています。  


```python
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import unicodedata
import math
import string
```


```python
# 記号文字は分析をするにあたって邪魔になるため、記号を取り除く関数を定義します。
# 下のYahooNews関数で使用します。
def symbol_removal(soup):
    soup = unicodedata.normalize("NFKC", soup)
    exclusion = "「」『』【】《》≪≫、。・◇◆" + "\n" + "\r" + "\u3000" # 除去する記号文字を指定
    soup = soup.translate(str.maketrans("", "", string.punctuation  + exclusion))
    return soup


# Yahooニュースをスクレイピングする関数です。
# 引数で指定した数の記事をとってきてデータフレームを返します。
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
        title_link = href_soup.find("a", class_="sc-eAyhxF")["href"] # title_link: 本文
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
```

では、YahooNews関数を利用して記事をデータフレームに格納しましょう


```python
df = YahooNews(1000)
```

    [====================] 497記事


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>category</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>菅首相 来年1月にも訪米検討</td>
      <td>国内</td>
      <td>菅義偉首相は3日の米大統領選の当選者と会談するため来年1月にも訪米する検討を始めた日本政府...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>燕14年ドラフト 6年で姿消す</td>
      <td>スポーツ</td>
      <td>2日にヤクルトは2014年ドラフト2位の風張ら7選手に戦力外通告 ヤクルトは2日近藤一樹投手...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>北海道また最多更新 街の声は</td>
      <td>地域</td>
      <td>UHB 北海道文化放送 感染拡大の猛威がとまりません北海道内では新たに1日の過去最多となる9...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GoTo食事券 県知事「改善を」</td>
      <td>地域</td>
      <td>新型コロナウイルスの感染拡大を受けて飲食業界を支援する国のGo To イートで兵庫県内で使...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SMBC コンビニATM手数料改定</td>
      <td>経済</td>
      <td>三井住友銀行は2日来年4月5日からコンビニの現金自動預払機ATMの手数料を改定すると発表し...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>492</th>
      <td>Zeebra、家族と別居報道</td>
      <td>エンタメ</td>
      <td>自分の不甲斐ない行いにより大切な家族を傷つけてしまった事を深く反省しております今後は家族ひと...</td>
    </tr>
    <tr>
      <th>493</th>
      <td>NYダウ大幅続落 650ドル安</td>
      <td>経済</td>
      <td>ニューヨーク共同週明け26日のニューヨーク株式市場のダウ工業株30種平均は大幅続落し前週末...</td>
    </tr>
    <tr>
      <th>494</th>
      <td>ももクロ高城 阪神D1にエール</td>
      <td>エンタメ</td>
      <td>阪神がドラフト1位で交渉権を獲得した近大佐藤輝明内野手21が大ファンだと公言しているももい...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>アント 史上最高額で上場へ</td>
      <td>経済</td>
      <td>上海共同中国の電子商取引最大手アリババグループ傘下で電子決済サービスアリペイを運営するアン...</td>
    </tr>
    <tr>
      <th>496</th>
      <td>所信表明 言い間違い6カ所</td>
      <td>国内</td>
      <td>菅義偉首相が26日の所信表明演説で新型コロナウイルス対策を巡り医療資源を重症者に重点化しま...</td>
    </tr>
  </tbody>
</table>
<p>497 rows × 3 columns</p>
</div>



### 工夫した点：  
- 引数を受け取って必要な数だけ記事をとってくることができるようにしました。
- 進捗が確認できるようにしました。(取って来る記事数が多いと、かなり時間がかかるので。)
- 引数に1000などの大きな数字を入れておくことで、Yahooニュースの記事を全て取得することができます。(yahooニュースは約500記事程度。)
- 記号を除去する関数も実装し、YahooNews関数に組み込んでいます。

### 妥協した点:   
- たまーに複数ページに渡って書かれている記事があるが、その場合は最初の1ページだけとってくる仕様にしております。
- コードがすごく長くなってしまいました。。もっと短くしたいです。
- 最初に全てのページの記事のリンクを取得するようにしているので、引数に指定した記事数が少なくても、最初に30秒程時間がかかりますし、その間は進捗も表示されません。


```python

```
