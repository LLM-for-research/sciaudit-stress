"""Тесты клиента к OpenAI-совместимому API (решение issue #12).

Сеть не трогается ни в одном тесте: HTTP подменяется. Главное, что здесь
сторожится, — ключ не должен утекать никуда, кроме заголовка запроса.
"""
import json

import pytest

from sciaudit.baselines import model_client as mc


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "api_key_env: SCIAUDIT_API_KEY\n"
        "request_defaults:\n"
        "  max_tokens: 128\n"
        "  temperature: 0.0\n"
        "  timeout_seconds: 5\n"
        "  max_retries: 3\n"
        "approved_models:\n"
        "  - approved-open-model\n",
        encoding="utf-8",
    )
    return path


def _config(config, **env):
    environ = {"SCIAUDIT_API_KEY": "sk-secret",
               "SCIAUDIT_BASE_URL": "https://example.invalid/v1",
               "SCIAUDIT_MODEL": "cfg-model", **env}
    return mc.load_config(config_path=config, environ=environ)


# --- .env -----------------------------------------------------------------------

def test_dotenv_is_parsed_and_quotes_are_stripped(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('# комментарий\nSCIAUDIT_MODEL="quoted-model"\n\nEMPTYLINE\n'
                        "SCIAUDIT_BASE_URL=https://host/v1\n", encoding="utf-8")
    environ = {}
    loaded = mc.load_dotenv(env_file, environ)
    assert loaded == {"SCIAUDIT_MODEL": "quoted-model",
                      "SCIAUDIT_BASE_URL": "https://host/v1"}


def test_real_environment_wins_over_dotenv(tmp_path):
    """В CI секреты приходят переменными; забытый .env не должен их перебивать."""
    env_file = tmp_path / ".env"
    env_file.write_text("SCIAUDIT_MODEL=from-file\n", encoding="utf-8")
    environ = {"SCIAUDIT_MODEL": "from-environment"}
    mc.load_dotenv(env_file, environ)
    assert environ["SCIAUDIT_MODEL"] == "from-environment"


def test_missing_dotenv_is_not_an_error(tmp_path):
    assert mc.load_dotenv(tmp_path / "nope.env", {}) == {}


# --- конфигурация ------------------------------------------------------------------

def test_endpoint_and_model_come_from_the_environment(config):
    """Привязки к бэкенду нет: vLLM отличается от внешнего API только адресом."""
    local = _config(config, SCIAUDIT_BASE_URL="http://localhost:8000/v1/",
                    SCIAUDIT_MODEL="local-model")
    remote = _config(config, SCIAUDIT_BASE_URL="https://api.vendor.example/v1",
                     SCIAUDIT_MODEL="remote-model")

    assert local["base_url"] == "http://localhost:8000/v1"   # хвостовой слэш снят
    assert remote["base_url"] == "https://api.vendor.example/v1"
    assert (local["max_tokens"], local["temperature"]) == \
        (remote["max_tokens"], remote["temperature"])


def test_request_defaults_come_from_the_config_file(config):
    settings = _config(config)
    assert settings["max_tokens"] == 128
    assert settings["temperature"] == 0.0
    assert settings["timeout_seconds"] == 5


def test_readiness_lists_exactly_what_is_missing(config):
    settings = mc.load_config(config_path=config, environ={})
    missing = mc.check_ready(settings)
    assert len(missing) == 3
    assert any("SCIAUDIT_BASE_URL" in m for m in missing)
    assert any("SCIAUDIT_MODEL" in m for m in missing)
    assert any("SCIAUDIT_API_KEY" in m for m in missing)


def test_approval_is_checked_against_the_model_not_the_backend(config):
    """§7.4 ограничивает саму модель, а не способ доступа к ней."""
    assert _config(config, SCIAUDIT_MODEL="approved-open-model")["approved"] is True
    assert _config(config, SCIAUDIT_MODEL="something-else")["approved"] is False
    # одна и та же модель одобрена независимо от того, локальный это эндпоинт или нет
    assert _config(config, SCIAUDIT_MODEL="approved-open-model",
                   SCIAUDIT_BASE_URL="http://localhost:8000/v1")["approved"] is True


def test_empty_approved_list_means_nothing_is_leaderboard_ready(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("api_key_env: SCIAUDIT_API_KEY\napproved_models: []\n", encoding="utf-8")
    settings = mc.load_config(config_path=path, environ={"SCIAUDIT_MODEL": "any"})
    assert settings["approved"] is False


# --- запрос ------------------------------------------------------------------------

def test_request_carries_the_key_in_the_header_and_is_deterministic(config, monkeypatch):
    seen = {}

    def fake_post(url, payload, api_key, timeout_seconds):
        seen.update(url=url, payload=payload, api_key=api_key, timeout=timeout_seconds)
        return {"choices": [{"message": {"content": "ответ модели"}}]}

    monkeypatch.setattr(mc, "_post", fake_post)
    assert mc.complete("промпт", _config(config)) == "ответ модели"

    assert seen["url"] == "https://example.invalid/v1/chat/completions"
    assert seen["api_key"] == "sk-secret"
    assert seen["payload"]["temperature"] == 0.0        # §6.1: воспроизводимость
    assert seen["payload"]["model"] == "cfg-model"
    assert seen["payload"]["messages"] == [{"role": "user", "content": "промпт"}]


def test_incomplete_setup_refuses_before_any_request(config, monkeypatch):
    monkeypatch.setattr(mc, "_post", lambda *a, **k: pytest.fail("запрос не должен уйти"))
    settings = mc.load_config(config_path=config, environ={})
    with pytest.raises(mc.ModelClientError, match="не готов"):
        mc.complete("промпт", settings)


def test_rate_limit_is_retried_then_succeeds(config, monkeypatch):
    import urllib.error
    calls = {"n": 0}

    def flaky(url, payload, api_key, timeout_seconds):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        return {"choices": [{"message": {"content": "наконец-то"}}]}

    monkeypatch.setattr(mc, "_post", flaky)
    assert mc.complete("промпт", _config(config), sleep=lambda s: None) == "наконец-то"
    assert calls["n"] == 3


def test_client_error_is_not_retried(config, monkeypatch):
    import urllib.error
    calls = {"n": 0}

    def unauthorized(url, payload, api_key, timeout_seconds):
        calls["n"] += 1
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(mc, "_post", unauthorized)
    with pytest.raises(mc.ModelClientError, match="401"):
        mc.complete("промпт", _config(config), sleep=lambda s: None)
    assert calls["n"] == 1


def test_malformed_response_is_reported_clearly(config, monkeypatch):
    monkeypatch.setattr(mc, "_post", lambda *a, **k: {"unexpected": True})
    with pytest.raises(mc.ModelClientError, match="choices"):
        mc.complete("промпт", _config(config))


def test_error_messages_never_contain_the_key(config, monkeypatch):
    import urllib.error
    monkeypatch.setattr(mc, "_post", lambda *a, **k: (_ for _ in ()).throw(
        urllib.error.HTTPError("u", 403, "Forbidden", {}, None)))
    with pytest.raises(mc.ModelClientError) as excinfo:
        mc.complete("промпт", _config(config), sleep=lambda s: None)
    assert "sk-secret" not in str(excinfo.value)


# --- reasoning-модели ----------------------------------------------------------------
#
# У reasoning-модели рассуждение списывается из того же бюджета max_tokens, что и
# ответ. При тесном лимите она думает до упора, content приходит пустым, и без
# разбора finish_reason это выглядит как поломка эндпоинта. Так и было на первом
# живом прогоне: лог обвинял эндпоинт, а виноват был лимит в 512 токенов.

def test_truncated_answer_blames_the_token_budget_not_the_endpoint(config, monkeypatch):
    monkeypatch.setattr(mc, "_post", lambda *a, **k: {"choices": [{
        "finish_reason": "length",
        "message": {"role": "assistant", "content": "", "reasoning": "думал-думал"},
    }]})
    with pytest.raises(mc.ModelClientError) as excinfo:
        mc.complete("промпт", _config(config))
    message = str(excinfo.value)
    assert "max_tokens" in message
    assert "обрезан" in message


def test_reasoning_without_content_is_reported_as_a_budget_problem(config, monkeypatch):
    monkeypatch.setattr(mc, "_post", lambda *a, **k: {"choices": [{
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "", "reasoning": "только рассуждение"},
    }]})
    with pytest.raises(mc.ModelClientError, match="max_tokens"):
        mc.complete("промпт", _config(config))


def test_reasoning_field_is_ignored_when_content_is_present(config, monkeypatch):
    monkeypatch.setattr(mc, "_post", lambda *a, **k: {"choices": [{
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": '{"verdict": "warranted"}',
                    "reasoning": "рассуждение в ответ не идёт"},
    }]})
    assert mc.complete("промпт", _config(config)) == '{"verdict": "warranted"}'


# --- seed ----------------------------------------------------------------------------

def test_seed_is_sent_when_configured(tmp_path, monkeypatch):
    path = tmp_path / "models.yaml"
    path.write_text("api_key_env: SCIAUDIT_API_KEY\n"
                    "request_defaults:\n  max_tokens: 128\n  seed: 7\n"
                    "approved_models: []\n", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(mc, "_post", lambda url, payload, key, t: seen.update(payload=payload)
                        or {"choices": [{"message": {"content": "ok"}}]})
    mc.complete("промпт", _config(path))
    assert seen["payload"]["seed"] == 7


def test_seed_is_omitted_when_not_configured(config, monkeypatch):
    seen = {}
    monkeypatch.setattr(mc, "_post", lambda url, payload, key, t: seen.update(payload=payload)
                        or {"choices": [{"message": {"content": "ok"}}]})
    mc.complete("промпт", _config(config))
    assert "seed" not in seen["payload"]


# --- проверка настройки --------------------------------------------------------------

def test_check_does_not_touch_the_network(config, monkeypatch):
    monkeypatch.setattr(mc, "CONFIG_PATH", config)
    monkeypatch.setattr(mc, "load_dotenv", lambda *a, **k: {})
    monkeypatch.setattr(mc, "_post", lambda *a, **k: pytest.fail("--check не должен ходить в сеть"))
    monkeypatch.setenv("SCIAUDIT_API_KEY", "sk-secret")
    monkeypatch.setenv("SCIAUDIT_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("SCIAUDIT_MODEL", "approved-open-model")
    assert mc.main(["--check"]) == 0


def test_ping_fails_loudly_when_the_endpoint_is_unreachable(config, monkeypatch, capsys):
    import urllib.error
    monkeypatch.setattr(mc, "CONFIG_PATH", config)
    monkeypatch.setattr(mc, "load_dotenv", lambda *a, **k: {})
    monkeypatch.setattr(mc, "_post", lambda *a, **k: (_ for _ in ()).throw(
        urllib.error.URLError("соединение отклонено")))
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    monkeypatch.setenv("SCIAUDIT_API_KEY", "sk-secret")
    monkeypatch.setenv("SCIAUDIT_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("SCIAUDIT_MODEL", "approved-open-model")
    # Заполненный .env - ещё не работающий доступ: --check это пропустит, --ping нет.
    assert mc.main(["--ping"]) == 1
    assert "НЕ ГОТОВ" in capsys.readouterr().err


# --- связка с бейзлайнами -----------------------------------------------------------

def test_baselines_reject_api_and_command_together():
    from sciaudit.baselines.model_audit import resolve_model

    class Args:
        model_api = True
        model_command = "echo {}"
        model_name = "x"

    with pytest.raises(ValueError, match="взаимоисключающи"):
        resolve_model(Args())


def test_shell_command_path_is_unchanged_without_the_api_flag():
    from sciaudit.baselines.model_audit import resolve_model

    class Args:
        model_api = False
        model_command = "echo {}"
        model_name = "declared"

    assert resolve_model(Args()) == (None, "echo {}", "declared")


# --- сквозная проверка по HTTP ------------------------------------------------------

def test_round_trip_against_a_local_openai_compatible_server(config):
    """Настоящий HTTP-запрос к локальному серверу: ровно так выглядит vLLM."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["payload"] = json.loads(body)
            payload = json.dumps({"choices": [{"message": {"content": "живой ответ"}}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    port = server.server_address[1]

    settings = _config(config, SCIAUDIT_BASE_URL=f"http://127.0.0.1:{port}/v1",
                       SCIAUDIT_MODEL="local-model")
    try:
        assert mc.complete("аудируй этот claim", settings) == "живой ответ"
    finally:
        server.server_close()

    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-secret"
    assert seen["payload"]["model"] == "local-model"
    assert seen["payload"]["temperature"] == 0.0
