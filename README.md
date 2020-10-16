### dockerで構築しているので皆様のパソコンでもコードを実行できます。
0. ご自身のデバイスに合わせて、下記を事前にインストールお願いします。
- Docker(Windows 10 proffesional Edition, macOS)
- Docker Toolbox(Windows 10 home edition)

1. このgithubをクローンしましょう。実行したいディレクトリに移動したら下記のコードをターミナルに入力して実行します。
```
git clone https://github.com/adachi0919/analysis001.git
```
2. `dock`というディレクトリまで移動してします。
```
cd analysis001/dock/
```
3. dockerを起動します。
```
docker-compose up --build
```
4. dockerが起動して環境が構築されます。ターミナルに以下のような表示がされるかと思います。  
`toke=`以下のトークンをコピーします。  
今回の場合は`b7dc9350ac6ae40c5313bd0ef6b838aac2158d04cf1e3637`です。
```
dock (🐳 :work) :$ docker-compose up
WARNING: Found orphan containers (dalex1, dock_nlp3_1, dock_nlp_1, dock_nlp2_1, dock_scraping_1) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Starting dock_morphologicalanalysis_1 ... done
Attaching to dock_morphologicalanalysis_1
morphologicalanalysis_1  | [I 10:05:58.426 LabApp] JupyterLab extension loaded from /opt/anaconda3/lib/python3.8/site-packages/jupyterlab
morphologicalanalysis_1  | [I 10:05:58.426 LabApp] JupyterLab application directory is /opt/anaconda3/share/jupyter/lab
morphologicalanalysis_1  | [I 10:05:58.429 LabApp] Serving notebooks from local directory: /work
morphologicalanalysis_1  | [I 10:05:58.429 LabApp] The Jupyter Notebook is running at:
morphologicalanalysis_1  | [I 10:05:58.429 LabApp] http://66abead6e041:8888/?token=b7dc9350ac6ae40c5313bd0ef6b838aac2158d04cf1e3637
morphologicalanalysis_1  | [I 10:05:58.429 LabApp]  or http://127.0.0.1:8888/?token=b7dc9350ac6ae40c5313bd0ef6b838aac2158d04cf1e3637
morphologicalanalysis_1  | [I 10:05:58.429 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
morphologicalanalysis_1  | [W 10:05:58.435 LabApp] No web browser found: could not locate runnable browser.
morphologicalanalysis_1  | [C 10:05:58.436 LabApp]
morphologicalanalysis_1  |
morphologicalanalysis_1  |     To access the notebook, open this file in a browser:
morphologicalanalysis_1  |         file:///root/.local/share/jupyter/runtime/nbserver-1-open.html
morphologicalanalysis_1  |     Or copy and paste one of these URLs:
morphologicalanalysis_1  |         http://66abead6e041:8888/?token=b7dc9350ac6ae40c5313bd0ef6b838aac2158d04cf1e3637
```
5. googlechromeのURLを入力する欄に`http://localhost:8826/`と入力ししてエンターを押しましょう。  
6. `jupyter`と表示される画面に遷移しますので`Password or token:`に先ほどコピーしたトークンをペーストしましょう。
これで私と同じdocker環境に入ることができました。  
中にあるコードを実行してみましょう。
