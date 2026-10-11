import asyncio
import json
import socket
import threading
import time
import wave

import pytest
import uvicorn

from podcast_server.cli import main


def test_list_voices(voices_dir, capsys):
    assert main(["--list-voices"]) == 0
    names = {v["name"] for v in json.loads(capsys.readouterr().out)}
    assert {"michael", "emma"} <= names


def test_unknown_speaker_exits_2(voices_dir, tmp_path, capsys):
    script = tmp_path / "s.txt"
    script.write_text("bob: hello\n")
    assert main([str(script), "-o", str(tmp_path / "o.wav")]) == 2
    assert "unknown speaker 'bob'" in capsys.readouterr().err
    assert not (tmp_path / "o.wav").exists()


def test_out_of_range_cfg_exits_2(voices_dir, tmp_path, capsys):
    script = tmp_path / "s.txt"
    script.write_text("michael: hello\n")
    assert main([str(script), "-o", str(tmp_path / "o.wav"), "--cfg-scale", "9"]) == 2
    assert "cfg_scale" in capsys.readouterr().err


def test_missing_script_file_exits_2(voices_dir, tmp_path):
    assert main([str(tmp_path / "nope.txt"), "-o", str(tmp_path / "o.wav")]) == 2


def test_requires_script_and_out(voices_dir):
    with pytest.raises(SystemExit) as err:
        main([])
    assert err.value.code == 2


@pytest.fixture
def live_mock_server(voices_dir):
    """A real uvicorn server (mock engine) on a free loopback port."""
    from podcast_server.engine import PodcastEngine
    from podcast_server.main import create_app
    from podcast_server.worker import MockWorker

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    engine = PodcastEngine(MockWorker, engine_name="mock", model_label="mock", idle_unload_secs=0)
    server = uvicorn.Server(uvicorn.Config(create_app(engine=engine), host="127.0.0.1",
                                           port=port, log_level="warning"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_remote_mode_writes_the_wav_and_a_summary(live_mock_server, tmp_path, capsys):
    script = tmp_path / "s.txt"
    script.write_text("# promo\nmichael: Hello there.\nemma: Hi.\n")
    out = tmp_path / "out" / "o.wav"
    assert main([str(script), "-o", str(out), "--server", live_mock_server, "--seed", "5"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["seed"] == 5 and summary["speakers"] == ["michael", "emma"]
    assert summary["truncated"] is False and summary["sample_rate"] == 24000
    with wave.open(str(out)) as w:
        assert abs(w.getnframes() / 24000 - summary["duration_secs"]) < 1e-3


def test_remote_mode_surfaces_server_errors(live_mock_server, tmp_path, capsys, monkeypatch):
    from podcast_server import cli

    # Bypass the CLI's local validation so the SERVER is the one refusing.
    script = tmp_path / "s.txt"
    script.write_text("michael: hi\n")
    monkeypatch.setattr(cli, "plan", lambda turns, voices, settings: _bogus_plan())
    assert main([str(script), "-o", str(tmp_path / "o.wav"), "--server", live_mock_server]) == 1
    assert "server answered 400" in capsys.readouterr().err


def _bogus_plan():
    from podcast_server.script import Plan, Turn

    return Plan(turns=(Turn("nobody", "hi"),), speakers=("nobody",), roles=(0,),
                total_chars=2, max_new_tokens=10)
