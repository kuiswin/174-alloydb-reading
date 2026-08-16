# AI Vector Reading Memos (AlloyDB Omni × ScaNN)

Google Cloud 実践検証シリーズ【第174弾】のソースコード実体です。

Google AlloyDB Omni（pgvector / ScaNN / pg_bigm / pg_trgm）× Ollama (ローカルAI) による、完全オフライン・クラウド費用ゼロで動作する高機能セマンティック読書メモ検索検証環境です。

## 📖 詳しい解説・チュートリアル
本リポジトリの設計思想、ローカル検証（Docker Compose / AlloyDB Omni + Ollama）、および Google Cloud 本番デプロイ手順の詳細解説は、Qiita および技術ブログにて公開しています：

👉 **Qiita 記事一覧**: [https://qiita.com/kuis](https://qiita.com/kuis)  
👉 **Author Blog**: [https://kuis.win](https://kuis.win)

---

## 🚀 クイックスタート (ローカル検証)

```bash
# コンテナ起動 (AlloyDB Omni + Ollama + Go API)
docker compose up -d

# ブラウザでアクセス
open http://localhost/
```

---

* 📜 **License**: MIT License (Copyright (c) 2026 kuiswin)
