# Claude Code Proxy Server (Gateway)

A high-performance, open-source, and modular proxy server designed to intercept Claude Code CLI/VSCode extension API requests (`/v1/messages`) and route them seamlessly to third-party LLM providers (NVIDIA NIM, OpenRouter, Gemini) or local endpoints (LM Studio, Ollama).

---

## 🏛️ Shared, Core & Atomic Yazılım Mimarisi

Proje, S.O.L.I.D ilkelerine ve modüler **Shared, Core, Atomic** tasarım deseni mimarisine uygun olarak yeniden yapılandırılmıştır.

```
claude-code-proxy/
├── shared/             # Stateles Katman (Pydantic şemaları, SSE yardımcıları, Hata sınıfları)
│   ├── schemas/        # Anthropic & OpenAI payload/delta modelleri
│   ├── utils/          # Pure SSE formatlayıcı ve ayrıştırıcılar
│   └── exceptions.py   # Tip güvenli özel istisna hiyerarşisi
├── atomic/             # Durumlu Micro-Bileşen Katmanı (Single Responsibility)
│   ├── parsers/        # ThinkingParser (<think> etiketleri) & HeuristicToolParser
│   └── guards/         # PreflightGuard, TokenBudgetGuard, SubagentGuard, StreamGuard
├── core/               # Orkestrasyon & İş Mantığı Katmanı
│   ├── gateway.py      # FastAPI route handler'ları (/v1/messages, /v1/models)
│   ├── router/         # ModelSelector, CircuitBreaker & DynamicRateLimiter
│   └── transformer/    # StreamEngine (Akış orkestratörü)
├── cli/                # Terminal arayüzü ve oturum yönetimi (session.py, main.py)
├── config/             # Ayarlar (.env) ve model kataloğu (models.yaml)
├── api/                # Hermes Gate Dashboard, MCP Server (api/mcp.py) & 0-token local mock interceptor
├── messaging/          # Telegram & Discord bot uzaktan yönetim entegrasyonu
├── mcp_server.py       # Hermes Agent Stdio MCP sunucusu
└── tests/              # 99 adet pytest birim testi
```

---

## 🛠️ Temel Katmanlar ve Özellikler

### 1. Shared Katmanı (`shared/`)
- **Durumsuz (Stateless) Yapı:** Hiçbir durum saklamaz. Yalnızca veri transfer nesnelerini (DTO) ve saf yardımcı fonksiyonları barındırır.
- **Güçlü Hata Hiyerarşisi (`exceptions.py`):** `CircuitOpenError`, `RateLimitExceededError`, `ContextOverflowError`, `SubagentPolicyViolationError`.

### 2. Atomic Katmanı (`atomic/`)
- **Tek Sorumluluklu Mikro Bileşenler:** Akış modunda veri işleyen durumlu (stateful) yapılar.
- **Thinking & Heuristic Tool Parsers:** `<think>` akıl yürütme etiketlerini ve metin içi markdown/JSON komutlarını (`/graphify`, `pnpm dev`, `git status`) `tool_use` event'lerine dönüştürür.
- **Güvenlik & Bütçe Korumaları:** `PreflightGuard` (1-token erişilebilirlik probe'u), `TokenBudgetGuard` (tiktoken ile context clipping), `SubagentGuard` (`run_in_background=False` zorlaması) ve `StreamGuard` (timeout & stall detector).

### 3. Core Katmanı (`core/`) & API / MCP Sunucusu
- **İş Mantığı & Orkestrasyon:**
  - **`StreamEngine`:** Upstream yanıt akışlarını alıp sırasıyla Atomic parser ve guard'lardan geçirir.
  - **`ModelSelector` & Resilience:** Models kataloğuna göre primary ve fallback modelleri yönetir, Circuit Breaker ve Rate Limiter durumlarına göre otomatik model değiştirir.
  - **`core.gateway`:** FastAPI `/v1/messages`, `/v1/models` ve `/v1/messages/count_tokens` uç noktaları.
  - **Sağlayıcı Bazlı Canlı RPM & TPM Metrikleri:** 12 LLM sağlayıcısının tamamı için dakikalık istek (RPM) ve token (TPM) kullanımı 60s pencerede canlı takip edilir.
  - **Hermes Agent İçin MCP (Model Context Protocol) Sunucusu (`api/mcp.py` & `mcp_server.py`):** HTTP/SSE ve Stdio JSON-RPC 2.0 üzerinden model listeleme (`get_models`), model hedefi değiştirme (`set_model_mapping`), sistem/sağlayıcı ayarları okuma/güncelleme (`get_system_config`, `update_system_config`), canlı metrikler (`get_metrics`) ve devre kesici yönetimi (`control_circuit_breaker`).

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleyin (`uv`)

[astral-uv](https://github.com/astral-sh/uv) kurulu olduğundan emin olun:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 2. Konfigürasyon (.env)

Örnek `.env` dosyasını kopyalayın ve API anahtarlarınızı girin:
```bash
cp .env.example .env
```

### 3. Sunucuyu Başlatın

FastAPI gateway sunucusunu başlatmak için:
```bash
uv run python cli/main.py start --port 8090
```
veya doğrudan:
```bash
uv run python server.py
```

### 4. Stdio MCP Sunucusunu Başlatın (Hermes Agent Entegrasyonu)

```bash
uv run python mcp_server.py
```

---

## 💻 Claude Code CLI Entegrasyonu

Claude Code CLI isteklerini yerel proxy sunucusuna yönlendirmek için:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8090"
export ANTHROPIC_AUTH_TOKEN="dummy_token"

# Teşhis ve Doğrulama
claude doctor

# Etkileşimli kodlama oturumunu başlatın
claude
```

---

## 📊 Dashboard ve Teşhis

- **Hermes Gate Control Dashboard UI:** `http://localhost:8090/dashboard`
- **MCP Sunucu Uç Noktaları:** `http://localhost:8090/mcp` & `http://localhost:8090/mcp/sse`
- **Sistem Sağlık Teşhisi (Doctor):**
  ```bash
  uv run python cli/main.py doctor
  ```
- **Birim Testlerini Çalıştırın (99 Test):**
  ```bash
  uv run pytest -v
  ```
- **Kod Kalitesi Denetimi (Ruff):**
  ```bash
  uv run ruff check .
  ```
