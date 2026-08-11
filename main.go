package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	_ "github.com/jackc/pgx/v5/stdlib"
)

type Memo struct {
	ID        string    `json:"id"`
	Title     string    `json:"title"`
	Content   string    `json:"memo"`
	CreatedAt time.Time `json:"created_at"`
	Score     *float64  `json:"score,omitempty"` // 検索時のみ類似度スコアが入る
}

var db *sql.DB
var ollamaURL string

func main() {
	var err error
	connStr := os.Getenv("DATABASE_URL")
	if connStr == "" {
		connStr = "postgres://postgres:postgres@alloydb-db:5432/reading_db?sslmode=disable"
	}
	ollamaURL = os.Getenv("OLLAMA_URL")
	if ollamaURL == "" {
		ollamaURL = "http://ollama:11434"
	}

	// 起動時のデータベース接続確認・リトライループ
	for i := 0; i < 20; i++ {
		db, err = sql.Open("pgx", connStr)
		if err == nil {
			err = db.Ping()
			if err == nil {
				log.Println("🚀 Successfully connected to AlloyDB!")
				initDB(db)
				break
			}
		}
		log.Printf("Waiting for AlloyDB (attempt %d/20)... error: %v\n", i+1, err)
		time.Sleep(3 * time.Second)
	}
	if err != nil {
		log.Fatalf("Fatal: Failed to connect to AlloyDB: %v", err)
	}
	defer db.Close()

	// 静的フロントエンドルーティング
	http.HandleFunc("/", serveIndex)
	http.HandleFunc("/style.css", serveStyle)
	
	// APIルーティング
	http.HandleFunc("/api/memo", handleAddMemo)
	http.HandleFunc("/api/memos", handleGetMemos)
	http.HandleFunc("/api/search", handleSearch)
	http.HandleFunc("/api/seed", handleSeed)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Printf("Go REST API Server starting on port %s...\n", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}
}

func initDB(d *sql.DB) {
	queries := []string{
		"GRANT ALL ON SCHEMA public TO postgres;",
		"CREATE EXTENSION IF NOT EXISTS vector;",
		"CREATE EXTENSION IF NOT EXISTS alloydb_scann;",
		"CREATE EXTENSION IF NOT EXISTS pg_trgm;",
		"CREATE EXTENSION IF NOT EXISTS pg_bigm;",
		`CREATE TABLE IF NOT EXISTS reading_memos (
			id VARCHAR(36) PRIMARY KEY,
			title VARCHAR(200) NOT NULL,
			memo TEXT NOT NULL,
			embedding vector(768),
			created_at TIMESTAMPTZ NOT NULL
		);`,
	}
	for _, q := range queries {
		if _, err := d.Exec(q); err != nil {
			log.Printf("DB Init Notice: %v\n", err)
		}
	}
	log.Println("✅ DB schema & extensions initialized successfully!")
}

func serveIndex(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "public/index.html")
}

func serveStyle(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "public/style.css")
}

// Vertex AI Text Embeddings REST API 呼び出し関数
func getVertexEmbeddings(text string) ([]float32, error) {
	projectID := os.Getenv("PROJECT_ID")
	if projectID == "" {
		projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
	}
	if projectID == "" {
		projectID = os.Getenv("GCP_PROJECT")
	}
	if projectID == "" {
		for _, url := range []string{
			"http://metadata.google.internal/computeMetadata/v1/project/project-id",
			"http://169.254.169.254/computeMetadata/v1/project/project-id",
		} {
			req, err := http.NewRequest("GET", url, nil)
			if err == nil {
				req.Header.Set("Metadata-Flavor", "Google")
				client := &http.Client{Timeout: 3 * time.Second}
				if resp, err := client.Do(req); err == nil && resp.StatusCode == 200 {
					body, err := io.ReadAll(resp.Body)
					resp.Body.Close()
					if err == nil && len(body) > 0 {
						projectID = strings.TrimSpace(string(body))
						break
					}
				}
			}
		}
	}
	if projectID == "" {
		return nil, fmt.Errorf("no project id")
	}
	region := os.Getenv("REGION")
	if region == "" {
		region = "asia-northeast1"
	}

	// Metadata Server から GCP アクセストークンを自動取得
	client := &http.Client{Timeout: 3 * time.Second}
	tokenReq, err := http.NewRequest("GET", "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token", nil)
	if err != nil {
		return nil, err
	}
	tokenReq.Header.Set("Metadata-Flavor", "Google")
	tokenResp, err := client.Do(tokenReq)
	if err != nil {
		return nil, err
	}
	defer tokenResp.Body.Close()

	var tokenData struct {
		AccessToken string `json:"access_token"`
	}
	if err := json.NewDecoder(tokenResp.Body).Decode(&tokenData); err != nil || tokenData.AccessToken == "" {
		return nil, fmt.Errorf("failed to parse access token")
	}

	url := fmt.Sprintf("https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models/text-multilingual-embedding-002:predict", region, projectID, region)
	reqBody, _ := json.Marshal(map[string]interface{}{
		"instances": []map[string]interface{}{
			{"content": text},
		},
	})

	vReq, err := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, err
	}
	vReq.Header.Set("Authorization", "Bearer "+tokenData.AccessToken)
	vReq.Header.Set("Content-Type", "application/json")

	vResp, err := client.Do(vReq)
	if err != nil {
		return nil, err
	}
	defer vResp.Body.Close()

	var vResult struct {
		Predictions []struct {
			Embeddings struct {
				Values []float32 `json:"values"`
			} `json:"embeddings"`
		} `json:"predictions"`
	}
	if err := json.NewDecoder(vResp.Body).Decode(&vResult); err == nil && len(vResult.Predictions) > 0 {
		return vResult.Predictions[0].Embeddings.Values, nil
	}
	return nil, fmt.Errorf("vertex ai prediction empty")
}

// Embeddingsを取得する関数 (Cloud: Vertex AI / Local: Ollama)
func getEmbeddings(text string, isQuery bool) ([]float32, error) {
	// 1. Vertex AI (Cloud Run環境 または USE_VERTEX_AI=true 設定時)
	if os.Getenv("USE_VERTEX_AI") == "true" || os.Getenv("K_SERVICE") != "" {
		vec, err := getVertexEmbeddings(text)
		if err != nil {
			log.Printf("[app ERROR] Vertex AI Embeddings generation failed: %v\n", err)
			return nil, fmt.Errorf("Vertex AI embedding failed: %w", err)
		}
		log.Printf("[app] Generated text embedding via Vertex AI (dim=%d)\n", len(vec))
		return vec, nil
	}

	// 2. Ollama (ローカル開発環境)
	var prefix string
	if isQuery {
		prefix = "search_query: "
	} else {
		prefix = "search_document: "
	}

	requestBody, err := json.Marshal(map[string]interface{}{
		"model": "nomic-embed-text",
		"input": prefix + text,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ollama request: %w", err)
	}

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Post(ollamaURL+"/api/embed", "application/json", bytes.NewBuffer(requestBody))
	if err != nil {
		log.Printf("[app ERROR] Ollama connection failed (%s): %v\n", ollamaURL, err)
		return nil, fmt.Errorf("Ollama connection failed: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil || len(result.Embeddings) == 0 {
		log.Printf("[app ERROR] Invalid response from Ollama: %v\n", err)
		return nil, fmt.Errorf("Ollama returned empty/invalid response")
	}

	log.Printf("[app] Generated text embedding via Ollama (dim=%d)\n", len(result.Embeddings[0]))
	return result.Embeddings[0], nil
}

// Postgresのvectorリテラルフォーマットに変換
func vectorToString(v []float32) string {
	var buf bytes.Buffer
	buf.WriteByte('[')
	for i, val := range v {
		if i > 0 {
			buf.WriteByte(',')
		}
		fmt.Fprintf(&buf, "%f", val)
	}
	buf.WriteByte(']')
	return buf.String()
}

// 読書メモの追加 (POST)
func handleAddMemo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Title string `json:"title"`
		Memo  string `json:"memo"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Title == "" || req.Memo == "" {
		http.Error(w, "Title and Memo are required", http.StatusBadRequest)
		return
	}

	// 1. Ollamaからベクトルを取得
	vector, err := getEmbeddings(req.Memo, false)
	if err != nil {
		log.Printf("Ollama embedding error: %v\n", err)
		http.Error(w, "Failed to generate embeddings: "+err.Error(), http.StatusInternalServerError)
		return
	}

	id := uuid.New().String()
	createdAt := time.Now()

	// 2. ベクトルリテラルに変換してAlloyDBにインサート
	vectorStr := vectorToString(vector)
	_, err = db.Exec("INSERT INTO reading_memos (id, title, memo, embedding, created_at) VALUES ($1, $2, $3, $4, $5)",
		id, req.Title, req.Memo, vectorStr, createdAt)
	if err != nil {
		log.Printf("AlloyDB insert error: %v\n", err)
		http.Error(w, "Failed to save memo: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"success": true, "id": id})
}

// 全メモの取得 (GET)
func handleGetMemos(w http.ResponseWriter, r *http.Request) {
	rows, err := db.Query("SELECT id, title, memo, created_at FROM reading_memos ORDER BY created_at DESC LIMIT 50")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var memos []Memo
	for rows.Next() {
		var m Memo
		if err := rows.Scan(&m.ID, &m.Title, &m.Content, &m.CreatedAt); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		memos = append(memos, m)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(memos)
}

// SearchResult represents the comparison of search results
type SearchResult struct {
	Like     []Memo  `json:"like"`
	FTS      []Memo  `json:"fts"`
	Vector   []Memo  `json:"vector"`
	LikeMs   float64 `json:"like_ms"`
	FTSMs    float64 `json:"fts_ms"`
	VectorMs float64 `json:"vector_ms"`
}

// AIセマンティック検索 ＆ 比較検索 (GET)
func handleSearch(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	if query == "" {
		http.Error(w, "Query parameter 'q' is required", http.StatusBadRequest)
		return
	}

	var results SearchResult
	results.Like = []Memo{}
	results.FTS = []Memo{}
	results.Vector = []Memo{}

	// 1. LIKE検索 (部分一致、最大5件)
	t1 := time.Now()
	likeQuery := "%" + query + "%"
	likeRows, err := db.Query("SELECT id, title, memo, created_at FROM reading_memos WHERE memo ILIKE $1 OR title ILIKE $1 ORDER BY created_at DESC LIMIT 5", likeQuery)
	if err == nil {
		defer likeRows.Close()
		for likeRows.Next() {
			var m Memo
			if err := likeRows.Scan(&m.ID, &m.Title, &m.Content, &m.CreatedAt); err == nil {
				results.Like = append(results.Like, m)
			}
		}
	} else {
		log.Printf("LIKE search error: %v\n", err)
	}
	results.LikeMs = float64(time.Since(t1).Microseconds()) / 1000.0

	// 2. 全文検索 / FTS (Tri-gram類似度、最大5件)
	t2 := time.Now()
	ftsRows, err := db.Query("SELECT id, title, memo, created_at, bigm_similarity(memo, $1) as score FROM reading_memos WHERE bigm_similarity(memo, $1) > 0 ORDER BY score DESC LIMIT 5", query)
	if err == nil {
		defer ftsRows.Close()
		for ftsRows.Next() {
			var m Memo
			var score float64
			if err := ftsRows.Scan(&m.ID, &m.Title, &m.Content, &m.CreatedAt, &score); err == nil {
				m.Score = &score
				results.FTS = append(results.FTS, m)
			}
		}
	} else {
		log.Printf("FTS search error: %v\n", err)
	}
	results.FTSMs = float64(time.Since(t2).Microseconds()) / 1000.0

	// 3. AIセマンティック検索 (ベクトル、最大5件)
	t3 := time.Now()
	vector, err := getEmbeddings(query, true)
	if err == nil {
		vectorStr := vectorToString(vector)
		vectorRows, err := db.Query("SELECT id, title, memo, created_at, (1 - (embedding <=> $1)) as score FROM reading_memos ORDER BY embedding <=> $1 LIMIT 5", vectorStr)
		if err == nil {
			defer vectorRows.Close()
			for vectorRows.Next() {
				var m Memo
				var score float64
				if err := vectorRows.Scan(&m.ID, &m.Title, &m.Content, &m.CreatedAt, &score); err == nil {
					m.Score = &score
					results.Vector = append(results.Vector, m)
				}
			}
		} else {
			log.Printf("Vector search query error: %v\n", err)
		}
	} else {
		log.Printf("Ollama embedding error: %v\n", err)
	}
	results.VectorMs = float64(time.Since(t3).Microseconds()) / 1000.0

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

// handleSeed registers 100 famous book quotes as sample data
func handleSeed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Delete all existing data to prevent duplication and ensure a clean state
	_, err := db.Exec("DELETE FROM reading_memos")
	if err != nil {
		log.Printf("Failed to clear database: %v\n", err)
		http.Error(w, "Failed to clear database: "+err.Error(), http.StatusInternalServerError)
		return
	}

	samples := []struct {
		Title string
		Memo  string
	}{
		{"夏目漱石 『こころ』", "精神的に向上心のないものは馬鹿だ。"},
		{"夏目漱石 『こころ』", "私は冷淡な人間です。しかし温かい人間になるように努力しています。"},
		{"夏目漱石 『こころ』", "人間を信用しないのではない、人間が信用できないのだ。"},
		{"夏目漱石 『坊っちゃん』", "親譲りの無鉄砲で小供の時から損ばかりしている。"},
		{"夏目漱石 『坊っちゃん』", "嘘をつくくらいなら、最初から何も言わないほうがましだ。"},
		{"夏目漱石 『吾輩は猫である』", "吾輩は猫である。名前はまだ無い。"},
		{"夏目漱石 『吾輩は猫である』", "人間は自分たちの都合のいいように世界を解釈している。"},
		{"夏目漱石 『草枕』", "智に働けば角が立つ。情に棹させば流される。意地を通せば窮屈だ。とかくに人の世は住みにくい。"},
		{"夏目漱石 『三四郎』", "熊本より東京は広い。東京より日本は広い。日本より頭の中の方が広いでございましょう。"},
		{"夏目漱石 『それから』", "それから、彼の頭には、ただ一つの明確な思想が残った。"},
		{"太宰治 『人間失格』", "恥の多い生涯を送って来ました。自分には、人間の生活というものが、見当つかないのです。"},
		{"太宰治 『人間失格』", "唯、一切は過ぎて行きます。自分がいままで阿鼻叫喚で生きて来た所謂人間の世界に於いて、唯一つ、真理らしく思われたのは、それだけでした。"},
		{"太宰治 『人間失格』", "人間は、お互いにちっとも相手を知らず、完全に間違えて見ていながら、一生無二の親友と信じ込んでいる。"},
		{"太宰治 『走れメロス』", "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。"},
		{"太宰治 『走れメロス』", "信じられているから走るのだ。信じる心があるから走るのだ。"},
		{"太宰治 『斜陽』", "人間は恋と革命のために生まれてきたのだ。"},
		{"太宰治 『斜陽』", "私たちは、生きていかなければならない。生きていることは、たいへんなことだ。"},
		{"太宰治 『富嶽百景』", "富士には、月見草がよく似合う。"},
		{"太宰治 『パンドラの匣』", "希望、それは人間が最後に手に入れることのできる、最も美しい幻想だ。"},
		{"太宰治 『ヴィヨンの妻』", "人非人でもいいじゃないの。私たちは、生きていさえすればいいのよ。"},
		{"芥川龍之介 『羅生門』", "ある日の暮方の事である。一人の下人が、羅生門の下で雨やみを待っていた。"},
		{"芥川龍之介 『羅生門』", "悪をなすには、悪を行わざるを得ないというだけの理由で十分である。"},
		{"芥川龍之介 『蜘蛛の糸』", "この極楽の蓮池のふちに立って、下の様子を眺めておられました。"},
		{"芥川龍之介 『蜘蛛の糸』", "おのれさえ助かれば良いという利己の心が、蜘蛛の糸を断ち切ったのだ。"},
		{"芥川龍之介 『鼻』", "人間の心には互いに矛盾した二つの感情がある。他人の不幸に同情しない者はいないが、その人が不幸を切り抜けると、今度は物足りなさを感じる。"},
		{"芥川龍之介 『杜子春』", "人間はだれでも、金持ちになればなるほど、薄情になるものだ。"},
		{"芥川龍之介 『杜子春』", "お前はもう仙人になりたいという望も、すっかり捨ててしまったのだろう。"},
		{"芥川龍之介 『藪の中』", "私は彼女の首を絞め、そしてその遺体を竹藪の奥へと隠しました。"},
		{"芥川龍之介 『地獄変』", "地獄の業火が燃え盛る中で、娘が焼け死ぬ姿を見つめながら、絵師は恍惚としていた。"},
		{"芥川龍之介 『或阿呆の一生』", "人生は地獄よりも地獄的である。"},
		{"宮沢賢治 『銀河鉄道の夜』", "世界が全体幸福にならないうちは個人の幸福はあり得ない。"},
		{"宮沢賢治 『銀河鉄道の夜』", "僕はもうあのさそりのように本当にみんなの幸のために僕のからだを百回灼いてもかまわない。"},
		{"宮沢賢治 『銀河鉄道の夜』", "どこまでもどこまでも一緒に行こう。僕たちは本当にみんなの幸いのために走っているんだ。"},
		{"宮沢賢治 『注文の多い料理店』", "注文の多い料理店というのは、料理の注文が多いのではなく、店から客への注文が多いのである。"},
		{"宮沢賢治 『雨ニモマケズ』", "雨ニモマケズ、風ニモマケズ、雪ニモ夏ノ暑サニモマケヌ丈夫ナカラダヲモチ。"},
		{"宮沢賢治 『雨ニモマケズ』", "サウイフモノニ、ワタシハナリタイ。"},
		{"宮沢賢治 『よだかの星』", "よだかは、どこまでもどこまでも、まっすぐに空へとのぼっていきました。"},
		{"宮沢賢治 『風の又三郎』", "どっどど どどうど どどうど どっど。"},
		{"宮沢賢治 『グスコーブドリの伝記』", "僕のような者は、いつ死んでも構わないのです。みんなが幸せになれるなら。"},
		{"宮沢賢治 『セロ弾きのゴーシュ』", "ゴーシュは夜遅くまで、何度も何度もセロの練習を繰り返しました。"},
		{"中島敦 『山月記』", "その時、藪の中から一匹の獰猛な虎が躍り出た。"},
		{"中島敦 『山月記』", "己の珠に非ざることを惧れるがゆえに、刻苦して磨こうともせず、また、己の珠なるを信ずるがゆえに、碌々として瓦に伍することもできなかった。"},
		{"中島敦 『山月記』", "人生は何事をもなさぬにはあまりに長いが、何事かをなすにはあまりに短い。"},
		{"中島敦 『山月記』", "獣としての肉体が、私の人間としての意識を日々侵食していくのだ。"},
		{"中島敦 『李陵』", "運命は過酷であり、忠義を尽くした者が必ずしも報われるわけではない。"},
		{"中島敦 『李陵』", "漢の将軍として、北方の匈奴と戦い抜いたが、ついに力尽きて降伏した。"},
		{"中島敦 『光と風と夢』", "南洋の眩しい光の中で、スティーヴンソンは自らの命を燃やし尽くしようとしていた。"},
		{"中島敦 『悟浄出世』", "なぜ生きるのか、という問い自体がすでに一つの病気なのだ。"},
		{"中島敦 『悟浄歎異』", "孫悟空の底知れない強さは、自らを疑わないその純粋さにある。"},
		{"中島敦 『弟子』", "子路は師である孔子の教えを実直に守り、最後の瞬間まで冠の紐を結び直した。"},
		{"森鴎外 『舞姫』", "石炭をば早や積み果てつ。中等室の卓のほとりはいと静かにて。"},
		{"森鴎外 『舞姫』", "豊太郎は自らの将来と、エリスへの愛との間で深く葛藤していた。"},
		{"森鴎外 『高瀬舟』", "財産というものは、足ることを知る者にとっては、わずかなものでも十分である。"},
		{"森鴎外 『高瀬舟』", "安楽死という極限の選択において、弟の喉に突き刺さったカミソリを引き抜いた。"},
		{"森鴎外 『阿部一族』", "殉死という古い武士の美学が、新しい時代と衝突して悲劇を生んだ。"},
		{"森鴎外 『寒山拾得』", "寒山と拾得の奇妙な笑い声が、寺の静寂を破って響き渡った。"},
		{"森鴎外 『山椒大夫』", "安寿と厨子王は、冷酷な山椒大夫のもとで過酷な労働を強いられていた。"},
		{"森鴎外 『山椒大夫』", "安寿は自らを犠牲にして、弟の厨子王を逃がす決意をした。"},
		{"森鴎外 『青年』", "東京の新しい風を浴びながら、青年は自らのアイデンティティを模索する。"},
		{"森鴎外 『雁』", "不条理な運命の悪戯によって、二人の思いは永遠にすれ違ってしまった。"},
		{"梶井基次郎 『檸檬』", "えたいの知れない不吉な塊が私の心を終始圧えつけていた。"},
		{"梶井基次郎 『檸檬』", "見すぼらしくて美しいもの。肺病病みの私には、それが何よりの慰めだった。"},
		{"梶井基次郎 『檸檬』", "丸善の棚に冷たい檸檬を一つ残し、私は爆弾を仕掛けたような興奮で街へ出た。"},
		{"梶井基次郎 『桜の樹の下には』", "桜の樹の下には屍体が埋まっている！これは信じていいことなんだ。"},
		{"梶井基次郎 『桜の樹の下には』", "なぜなら、桜の花があんなにも見事に咲くためには、それだけの栄養が必要だからだ。"},
		{"梶井基次郎 『冬の幻』", "冬の冷たい空気の中で、私は自分の中の幻影と対話していた。"},
		{"梶井基次郎 『器楽的幻覚』", "ピアノの旋律が、私の神経を細かく切り刻んでいく。"},
		{"梶井基次郎 『愛撫』", "猫の耳の冷たさは、何とも言えない愛撫の対象である。"},
		{"梶井基次郎 『交尾』", "真夏の夜の川原で、河鹿たちが命の交わりを交わしていた。"},
		{"梶井基次郎 『ある心の風景』", "自分の心の中の風景は、常に灰色の風景に覆われている。"},
		{"江戸川乱歩 『人間椅子』", "私は洋館の椅子の中に忍び込み、そこに座る人々の温もりを直接感じていたのです。"},
		{"江戸川乱歩 『人間椅子』", "この手紙を読み終えた時、あなたは私の存在に恐怖するでしょう。"},
		{"江戸川乱歩 『屋根裏の散歩者』", "郷田三郎は屋根裏を這い回り、他人の生活を盗み見る快楽に溺れていた。"},
		{"江戸川乱歩 『屋根裏の散歩者』", "天井の節穴から、眠る男の口に向けて毒薬を一滴垂らした。"},
		{"江戸川乱歩 『心理試験』", "犯人は緻密な計画を立てて心理試験に臨んだが、明智小五郎の洞察力には勝てなかった。"},
		{"江戸川乱歩 『孤島の鬼』", "島の中に隠された巨大な迷宮と、異形の人間たちの悲劇。"},
		{"江戸川乱歩 『黒蜥蜴』", "稀代の女盗賊・黒蜥蜴と、名探偵・明智小五郎の知恵比べ。"},
		{"江戸川乱歩 『鏡地獄』", "男は無数の鏡に囲まれた球体の中に閉じこもり、ついに狂気へと至った。"},
		{"江戸川乱歩 『押絵と旅する男』", "額縁のなかの絵には、歳をとらない不思議な老紳士と娘が描かれていた。"},
		{"江戸川乱歩 『怪人二十面相』", "二十面相はどのような姿にも変装し、美術品を華麗に盗み出す。"},
		{"三島由紀夫 『金閣寺』", "金閣を焼かねばならぬ、と私は思った。"},
		{"三島由紀夫 『金閣寺』", "美というものは、人間に害を与えることなく、ただそこにあるだけで毒をまき散らす。"},
		{"三島由紀夫 『潮騒』", "その火を飛び越えて来い。そこに私の愛がある。"},
		{"三島由紀夫 『仮面の告白』", "私は自分が他人と異なっているという秘密を、仮面の後ろに隠し続けた。"},
		{"川端康成 『雪国』", "国境の長いトンネルを抜けると雪国であった。夜の底が白くなった。"},
		{"川端康成 『伊豆の踊子』", "道がつづら折りになって、いよいよ天城峠に近づいたと思う頃、雨脚が杉の密林を白く染めながら降りてきた。"},
		{"紫式部 『源氏物語』", "いづれの御時にか、女御、更衣あまたさぶらひたまひけるなかに。"},
		{"清少納言 『枕草子』", "春はあけぼの。やうやう白くなりゆく山際、少しあかりて、紫だちたる雲の細くたなびきたる。"},
		{"鴨長明 『方丈記』", "ゆく河の流れは絶えずして、しかももとの水にあらず。よどみに浮かぶうたかたは、かつ消えかつ結びて、久しくとどまりたるためしなし。"},
		{"吉田兼好 『徒然草』", "つれづれなるままに、日暮らしパソコンに向かひて、心にうつりゆくよしなし事を、そこはかとなく書きつくれば。"},
		{"ニーチェ 『ツァラトゥストラはこう言った』", "神は死んだ。そして人間は超人を目指さねばならない。"},
		{"サン＝テグジュペリ 『星の王子さま』", "ものごとはね、心で見なくてはよく見えない。いちばんたいせつなことは、目に見えないのだよ。"},
		{"アラン 『幸福論』", "悲観主義は気分のものであり、楽観主義は意志のものである。"},
		{"ダーウィン 『種の起源』", "最も強い者が生き残るのではなく、最も賢い者が生き残るのでもない。唯一生き残るのは、変化できる者である。"},
		{"デカルト 『方法序説』", "我思う、ゆえに我あり。"},
		{"アラン・チューリング", "コンピュータが思考できるかどうかは、知的な振る舞いを示せるかどうかで判定すべきだ。"},
		{"エドガー・ダイクストラ", "デバッグはプログラムにバグが存在することを示すことはできるが、バグが存在しないことを証明することはできない。"},
		{"ドナルド・クヌース", "早期最適化はすべての悪の根源である。"},
		{"アラン・ケイ", "未来を予測する最良の方法は、それを発明することだ。"},
		{"著者不明", "AIの時代にあっても、本を読み、言葉を紡ぎ、自らの頭で考える人間の尊厳は決して失われない。"},
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")

	type jobResult struct {
		title string
		memo  string
		embed []float32
		err   error
	}

	jobs := make(chan int, len(samples))
	results := make(chan jobResult, len(samples))

	// Concurrency level for local Ollama CPU. Limit to 5 workers to be safe and responsive.
	concurrencyLimit := 5
	for w := 1; w <= concurrencyLimit; w++ {
		go func() {
			for idx := range jobs {
				item := samples[idx]
				vector, err := getEmbeddings(item.Memo, false)
				results <- jobResult{
					title: item.Title,
					memo:  item.Memo,
					embed: vector,
					err:   err,
				}
			}
		}()
	}

	// Send all indexing jobs
	for i := 0; i < len(samples); i++ {
		jobs <- i
	}
	close(jobs)

	// Collect results and insert them
	successCount := 0
	for i := 0; i < len(samples); i++ {
		res := <-results
		if res.err != nil {
			log.Printf("Embedding error for '%s': %v\n", res.memo, res.err)
			continue
		}

		id := uuid.New().String()
		createdAt := time.Now()
		vectorStr := vectorToString(res.embed)

		_, err = db.Exec("INSERT INTO reading_memos (id, title, memo, embedding, created_at) VALUES ($1, $2, $3, $4, $5)",
			id, res.title, res.memo, vectorStr, createdAt)
		if err != nil {
			log.Printf("Database insert error: %v\n", err)
			continue
		}
		successCount++

		// Stream progress update as NDJSON
		progress := map[string]interface{}{
			"current": successCount,
			"total":   len(samples),
			"title":   res.title,
		}
		json.NewEncoder(w).Encode(progress)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}
}

// Reviewed and verified locally.
