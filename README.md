# Claude Code Proxy Server (Gateway)

A lightweight, open-source, and modular proxy server designed to intercept Claude Code CLI/VSCode extension API requests and route them to OpenAI-compatible third-party endpoints (such as NVIDIA NIM, OpenRouter) or local endpoints (LM Studio, Ollama).

---

## 🚀 Son Yapılan Mimari Yenilikler (Resilient Multi-Model Router & Gateway)

Proje, basit bir proxy katmanından kendi kendini onarabilen, rate-limit duyarlı ve token bütçesi korumalı kurumsal seviyede bir **Resilient Multi-Model Router & Gateway** mimarisine dönüştürülmüştür.

### Kullanılan Teknolojiler & Kütüphaneler

- **Core & Server:** Python 3.14+, [FastAPI](https://fastapi.tiangolo.com/), Uvicorn (Asenkron yüksek performanslı HTTP/SSE Gateway)
- **Model Registry & Config:** [PyYAML](https://pyyaml.org/) (`config/models.yaml` dinamik model kataloğu)
- **Token Counting & Context Guard:** [tiktoken](https://github.com/openai/tiktoken) (`cl100k_base` & `o200k_base` tokenizer entegrasyonu)
- **HTTP Client & Async I/O:** `httpx` (Asenkron streaming ve custom header parsing)
- **Paket & Bağımlılık Yönetimi:** `uv` (Astral)
- **Test & Linting:** `pytest`, `pytest-asyncio`, `ruff`

---

## 🛠️ Entegre Edilen Yöntemler ve Bileşenler

### 1. Model Registry (`config/models.yaml`)
- `primary` ve `fallback_order` hiyerarşisiyle model zincirleri tanımlanır.
- Modeller için `context`, `max_output`, `rpm_limit`, `tpm_limit` ve capability etiketleri (`reasoning`, `tool-calling`, `coding`, `agentic`) saklanır.

### 2. Circuit Breaker (`router/circuit_breaker.py`)
- Her model için bağımsız `CLOSED → OPEN → HALF_OPEN` durum makinesi çalışır.
- **Parametreler:** `failure_threshold = 5`, `recovery_timeout = 60s`.
- 5 ardışık hatada devre açılır (`OPEN`) ve istekler yedek modellere yönlendirilir. 60s sonra `HALF_OPEN` modunda deneme yapılır.

### 3. Dynamic Rate Limit Parser (`router/rate_limit_parser.py`)
- Upstream API (NVIDIA NIM, OpenRouter) yanıtlarındaki HTTP başlıklarını (`x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-*`) anlık parse eder.
- **Headroom Politikası:** Kalan kota kapapasitenin %10'unun altına düştüğünde router otomatik olarak sıradaki sağlıklı fallback modele geçiş yapar.

### 4. Preflight Model Probe (`guards/preflight.py`)
- Kesintisiz çalışma prensibi doğrultusunda her istek öncesinde seçilen model 3 saniyelik hafif (1-token) bir probe isteği ile kontrol edilir. Erişilemez durumda devre kesici tetiklenir ve anında fallback modele geçilir.

### 5. Token Budget Guard (`guards/token_budget.py`)
- Upstream sağlayıcıların `400 Context Length Exceeded` hatası vermesini engeller.
- `tiktoken` ile mesaj token boyutlarını hesaplar. Llama/Mistral/Qwen/GLM modelleri için `o200k_base`, diğerleri için `cl100k_base` tokenizer kullanır.
- Limit aşımında System Prompt ve son Kullanıcı mesajını koruyarak eski konuşma turlarını otomatik kırpar (`smart_truncate`).

### 6. SSE Stream Guard (`guards/stream_guard.py`)
- Server-Sent Events akışlarını izler. 30 saniye boyunca chunk gelmemesi (timeout) veya 10 ardışık boş chunk durumunda akışın takıldığını algılar (`stall detection`) ve istemciye temiz bir Anthropic SSE error eventi döner.

---

## Features

1. **Sunucu Katmanı**: FastAPI & Uvicorn base, default olarak `http://localhost:8090` üzerinde çalışır.
2. **Çağrı Optimizasyonu (Local Mocking)**:
   - **Quota/Network Probes**: API erişilebilirlik kontrollerine yerel yanıt vererek token maliyetini sıfırlar.
   - **Title Generation**: Konuşma başlıklarını yerelde üretir.
   - **Prefix Detection**: Güvenlik komut ön eklerini anında çıkarır.
   - **Suggestion Mode**: Terminal otomatik tamamlama isteklerini mocklar.
   - **Filepath Extraction**: Dosya yollarını regex ile ayrıştırır.
3. **Model & Sağlayıcı Yönlendirme**:
   - `.env` ve `models.yaml` tabanlı yönlendirme (`nvidia_nim`, `open_router`, `lmstudio`, `ollama`).
4. **SSE (Server-Sent Events) Streaming Transformer**:
   - OpenAI tamamlamalarını Anthropic akış formatına dönüştürür.
   - Standard ve `<think>` / `</think>` akıl yürütme bloklarını Claude native `thinking` bloklarına çevirir.
   - Metin içi tool çağrılarını sezgisel (heuristic) JSON/markdown parser ile yapısal araç çıktılarına dönüştürür.
5. **Bot ve Uzaktan Yönetim**:
   - Telegram ve Discord botları üzerinden komut çalıştırma, başlık bazlı oturum takibi ve durum raporlama.
6. **Hermes Gate Control Dashboard (`/dashboard`)**:
   - Monokrom kontrol paneli üzerinden anlık istek metrikleri, model eşlemeleri, API anahtarı yönetimi, canlı log takibi ve **Router & Fallbacks** durum matrisi.

---

## Installation & Setup

[astral-uv](https://github.com/astral-sh/uv) kurulu olduğundan emin olun:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Bağımlılıkları yükleyin:
```bash
uv sync
```

### Environment Configuration

Örnek `.env` dosyasını kopyalayın ve API anahtarlarınızı girin:
```bash
cp .env.example .env
```

.env dosyası içerisinden veya `/dashboard` arayüzünden ayarları değiştirebilirsiniz:
- Upstream credentials (`NVIDIA_NIM_API_KEY`, `OPENROUTER_API_KEY` vb.).
- Hedef model atamaları (`MODEL_OPUS`, `MODEL_SONNET`, `MODEL_HAIKU`, `MODEL`).

---

## Run Server

FastAPI gateway sunucusunu başlatın:
```bash
uv run python cli/main.py start --port 8090
```

veya doğrudan:
```bash
uv run python server.py
```

### Check Diagnostics & Router Health

Sunucu teşhisini çalıştırmak için:
```bash
uv run python cli/main.py doctor
```

Veya taraıcıdan kontrol paneline erişin:
- **Dashboard:** `http://localhost:8090/dashboard`
- **Router Health Status API:** `http://localhost:8090/api/router-status`

---

## Connect Claude Code

Claude Code CLI isteklerini proxy sunucusuna yönlendirmek için:
```bash
export ANTHROPIC_BASE_URL="http://localhost:8090"
export ANTHROPIC_AUTH_TOKEN="dummy_token"

# Claude Code teşhisi
claude doctor

# Etkileşimli kodlama oturumunu başlat
claude
```

---

## Bot Administration Commands

### Telegram Bot Commands
- `/status`: Ayarları ve model haritalarını gösterir.
- `/set_model <key> <val>`: Model haritalarını anlık günceller.
- `/run <bash command>`: Çalışma alanında komut çalıştırır.

### Discord Bot Commands
- `!status`: Ayarları ve model haritalarını gösterir.
- `!set_model <key> <val>`: Model haritalarını anlık günceller.
- `!run <bash command>`: Komut çalıştırır.
- *Real-time session monitoring*: Düşünceleri ve araç yürütmelerini kanal başlıklarına yanıt olarak iletir.
