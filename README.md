# Claude Code Proxy Server (Gateway)

A high-performance, open-source, and modular proxy server designed to intercept Claude Code CLI/VSCode extension API requests (`/v1/messages`) and route them seamlessly to third-party LLM providers (NVIDIA NIM, OpenRouter, Gemini, Groq, DeepSeek, Mistral, Cerebras, Fireworks, Kimi) or local endpoints (LM Studio, Ollama, Llama.cpp).

---

## 🏛️ Shared, Core & Atomic Yazılım Mimarisi

Proje, S.O.L.I.D ilkelerine ve modüler **Shared, Core, Atomic** tasarım deseni mimarisine uygun olarak yapılandırılmıştır.

```
claude-code-proxy/
├── shared/             # Stateles Katman (Pydantic şemaları, SSE yardımcıları, Hata sınıfları)
│   ├── schemas/        # Anthropic & OpenAI payload/delta modelleri
│   ├── utils/          # Pure SSE formatlayıcı ve ayrıştırıcılar
│   └── exceptions.py   # Tip güvenli özel istisna hiyerarşisi
├── atomic/             # Durumlu Micro-Bileşen Katmanı (Single Responsibility)
│   ├── parsers/        # ThinkingParser (<think> etiketleri) & HeuristicToolParser
│   └── guards/         # PreflightGuard, TokenBudgetGuard, SubagentGuard, StreamGuard, SubagentPolicyEngine, NIMThrottleGuard
├── core/               # Orkestrasyon & İş Mantığı Katmanı
│   ├── gateway.py      # FastAPI route handler'ları (/v1/messages, /v1/models, SlidingWindowRateLimiter O(1))
│   ├── router/         # ModelSelector, CircuitBreaker (Async/Lock & Atomic Persistence), DailyTracker
│   └── transformer/    # StreamEngine (Akış orkestratörü)
├── cli/                # Terminal arayüzü ve oturum yönetimi (session.py, main.py)
├── config/             # Ayarlar (.env, subagent_policy.yaml) ve model kataloğu (models.yaml)
├── api/                # Dashboard, MCP Server (api/mcp.py), Prometheus Metrics (api/metrics.py) & Claude Settings Bridge (api/settings_manager.py)
├── messaging/          # Telegram & Discord bot uzaktan yönetim entegrasyonu (Sanitized prompt injection)
├── mcp_server.py       # Hermes Agent Stdio MCP sunucusu
└── tests/              # 150+ adet pytest birim, eşzamanlılık ve kontrat testi
```

---

## 🚀 Son Güncellemeler ve Mimari İyileştirmeler

### 1. Dayanıklılık ve Eşzamanlılık (Resilience & Concurrency Fixes)
- **Async & Thread-Safe CircuitBreaker (`core/router/circuit_breaker.py`):** `is_open()`, `trip_or_extend()`, `reset()`, `force_open()` async yapıya kavuşturulup `asyncio.Lock` ile yarıştırma koşullarına (race condition) karşı tam korumaya alındı.
- **O(1) Sliding Window Rate Limiter (`core/gateway.py`):** Dizi filtreleme O(n) işleminden `collections.deque` yapısına geçilerek O(1) akış kontrolü sağlandı. `time.time()` yerine `time.monotonic()` kullanılarak sistem saati değişimlerinden etkilenmeyen tutarlı zamanlama sağlandı.
- **Preflight Probe Yanlış Pozitif Düzeltmesi (`atomic/guards/preflight.py`):** 4xx istemci yanıtlarında Circuit Breaker tetiklenmesi önlendi (yalnızca loglama yapılır); sadece 5xx sunucu hataları Circuit Breaker sayacını artırır.
- **Thread-Safe Key Rotation (`providers/openai.py`):** Sağlayıcı API key havuzu `asyncio.Lock` ile korunarak concurrent isteklerde mükerrer key kullanımı engellendi.
- **Atomik Dosya Kaydı (`core/router/circuit_breaker.py`):** Breaker durumları `.tmp` dosyasına yazılıp `os.replace` ile atomik olarak kaydedilir, çökme anında YAML bozulması engellenir.

### 2. Docker Secrets Entegrasyonu (`docker-compose.yml` & `config/config.py`)
- Hassas API anahtarları (`NVIDIA_NIM_API_KEY`, `OPENROUTER_API_KEY`, `GATEWAY_AUTH_TOKEN`) öncelikli olarak `/run/secrets/` dizininden okunur; bulunamazsa `.env` değişkenlerine düşer.

### 3. Model-Spesifik Tokenizer Haritası (`atomic/guards/token_budget.py`)
- `MODEL_TOKENIZER_MAP` ile `cl100k_base`, `o200k_base`, `p50k_base` gibi tokenizer kodlamaları model ailelerine göre dinamik eşleştirilir.

### 4. Telegram Prompt Sanitizasyonu (`core/gateway.py`)
- Telegram üzerinden kuyruğa alınan komutlar HTML escape, 2000 karakter sınırlandırması ve yasaklı enjeksiyon dizilim süzgecinden geçirilir.

### 5. YAML-Driven Subagent Policy Engine (`atomic/guards/subagent_policy.py`)
- Subagent araç çalıştırma kuralları `config/subagent_policy.yaml` dosyasından okunur; `SubagentPolicyEngine` izin/engel kararlarını denetler ve denetim günlüğü tutar.

### 6. Claude Code Settings Bridge (`api/settings_manager.py`)
- `~/.claude.json` (Kullanıcı), `.claude.json` (Proje) ve Yerel ayarları birleştirir. `sync_proxy_to_claude` aracı ile CLI yönlendirmesi otomatik yapılandırılır.

### 7. Metrikler ve Canlı Sıcak Yükleme (Metrics & Hot-Reload)
- **Prometheus Metrikleri (`/metrics`):** `proxy_requests_total`, `proxy_active_concurrency`, `proxy_error_total`, `proxy_circuit_state`, `proxy_rate_limit_headroom`.
- **Sağlık Kontrolleri:** `/healthz` (Liveness) ve `/readyz` (Readiness).
- **SIGHUP Sıcak Yükleme (`server.py`):** Sunucuyu yeniden başlatmadan `.env` ve `models.yaml` ayarlarını anlık günceller.

---

## 🛠️ MCP (Model Context Protocol) Araç Kataloğu (16 MCP Tool)

Sunucu, Hem Hermes Agent hem de harici asistanlar için 16 adet idari MCP aracı sunar:
1. `get_models`: Sağlayıcı model kataloğunu ve aktif istemci eşleşmelerini listeler.
2. `set_model_mapping`: İstemci rumuzlarına target model ve fallback zinciri atar.
3. `get_system_config`: Sistem ayarlarını ve sağlayıcı limitlerini okur.
4. `update_system_config`: Sistem konfigürasyonunu günceller.
5. `get_metrics`: Canlı trafik ve RPM/TPM metriklerini sunar.
6. `get_throttle_metrics`: Gecikme ve throttle telemetry bilgilerini getirir.
7. `update_throttle_settings`: NIM throttle parametrelerini dinamik günceller.
8. `get_model_routing`: Yönlendirme matrisini getirir.
9. `control_circuit_breaker`: Circuit breaker state'ini manuel sıfırlar veya keser.
10. `get_subagent_policy`: Subagent kural politikasını getirir.
11. `set_subagent_policy`: Subagent kural politikasını günceller.
12. `get_subagent_decisions`: Subagent karar denetim geçmişini sunar.
13. `manage_prompt_queue`: Telegram/CLI komut kuyruğunu yönetir (`peek`, `inject`, `replace`, `clear`).
14. `get_claude_settings`: Birleştirilmiş Claude ayarlarını getirir.
15. `set_claude_setting`: Claude ayarlarını günceller.
16. `sync_proxy_to_claude`: Claude CLI ortamını yerel proxy sunucusuna eşlemler.

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleyin (`uv`)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 2. Konfigürasyon (.env)
```bash
cp .env.example .env
```

### 3. Sunucuyu Başlatın
```bash
uv run python server.py
```

### 4. Stdio MCP Sunucusunu Başlatın
```bash
uv run python mcp_server.py
```

---

## 📊 Dashboard ve Teşhis

- **Dashboard UI:** `http://localhost:8090/dashboard`
- **Prometheus Metrikleri:** `http://localhost:8090/metrics`
- **Sağlık Uç Noktaları:** `http://localhost:8090/healthz` ve `http://localhost:8090/readyz`
- **MCP Sunucu Uç Noktaları:** `http://localhost:8090/mcp` & `http://localhost:8090/mcp/sse`
- **Birim & Eşzamanlılık Testleri:**
  ```bash
  uv run pytest -v
  ```

