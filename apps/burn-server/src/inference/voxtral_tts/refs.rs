//! Voice-reference storage for zero-shot voice cloning.
//!
//! Direct port of `apps/vllm-server/voxtral_server/tts/refs.py`. Owns
//! the local catalog of uploaded reference audio + the user-supplied
//! transcript. The actual synthesis path is gated on Phase 3-E2 (the
//! ref-encoder forward pass); this module only handles persistence,
//! validation, and the upload/delete audit log.
//!
//! Layout on disk under `BURN_TTS_VOICE_REFS_DIR`:
//! - `<id>.wav` — re-encoded 24 kHz mono PCM16 audio
//! - `<id>.json` — sidecar with name, transcript, permission flag
//! - `_audit.log` — append-only log of every upload + delete
//!
//! Path-traversal safety: the client never controls the on-disk
//! filename. We always write to `<uuid>.wav`/`<uuid>.json`. The
//! `voice_id` lookup APIs accept only `[a-f0-9-]{8,64}` strings.

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceRef {
    pub id: String,
    pub name: String,
    pub ref_text: String,
    pub permission_confirmed: bool,
    pub sample_rate: u32,
    pub duration_secs: f64,
    pub created_at: f64,
    pub audio_path: String,
    pub sidecar_path: String,
}

/// Parameters for `save_ref`. Borrowed-style struct keeps the call site
/// tidy when many of the fields come from server config.
pub struct SaveRefArgs<'a> {
    pub audio_bytes: &'a [u8],
    pub audio_filename: &'a str,
    pub name: &'a str,
    pub ref_text: &'a str,
    pub permission_confirmed: bool,
    pub voice_refs_dir: &'a Path,
    pub max_audio_bytes: usize,
    pub min_duration_secs: f64,
    pub max_duration_secs: f64,
    pub require_permission: bool,
}

const AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "flac", "ogg", "m4a"];
const REF_TEXT_MAX_CHARS: usize = 1000;

fn name_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z0-9 _\-]{1,64}$").expect("name regex"))
}

fn voice_id_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[a-f0-9-]{8,64}$").expect("voice_id regex"))
}

fn sanitize_name(name: &str) -> Result<String> {
    let trimmed = name.trim();
    if !name_re().is_match(trimmed) {
        bail!("name must be 1-64 chars of [A-Za-z0-9 _-]");
    }
    Ok(trimmed.to_string())
}

fn sanitize_voice_id(voice_id: &str) -> Result<String> {
    if !voice_id_re().is_match(voice_id) {
        bail!("voice_ref_id must look like a UUID hex");
    }
    Ok(voice_id.to_string())
}

/// Persist a reference audio + transcript. Returns the new `VoiceRef`.
/// `voice_refs_dir` is created on demand. The caller has already
/// pulled the bytes out of the multipart upload; we own everything
/// else (re-encoding, sanitization, sidecar write, audit log).
///
/// IO: blocking. Always invoke from `tokio::task::spawn_blocking`.
pub fn save_ref(args: SaveRefArgs<'_>) -> Result<VoiceRef> {
    if args.require_permission && !args.permission_confirmed {
        bail!(
            "permission_confirmed must be true (you must confirm you have \
             permission from the speaker before uploading their voice)"
        );
    }
    if args.audio_bytes.is_empty() {
        bail!("audio file is empty");
    }
    if args.audio_bytes.len() > args.max_audio_bytes {
        bail!(
            "audio file is too large ({} > {} bytes)",
            args.audio_bytes.len(),
            args.max_audio_bytes
        );
    }

    let name = sanitize_name(args.name)?;
    let ref_text = args.ref_text.trim();
    if ref_text.is_empty() {
        bail!("ref_text must be a non-empty transcript of the audio");
    }
    if ref_text.chars().count() > REF_TEXT_MAX_CHARS {
        bail!("ref_text must be ≤ {REF_TEXT_MAX_CHARS} chars");
    }

    let src_suffix = Path::new(args.audio_filename)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    if !AUDIO_EXTENSIONS.contains(&src_suffix.as_str()) {
        bail!(
            "audio_filename must end in one of {:?} (got .{:?})",
            AUDIO_EXTENSIONS,
            src_suffix
        );
    }

    std::fs::create_dir_all(args.voice_refs_dir)?;

    let voice_id = uuid::Uuid::new_v4().simple().to_string();
    let src_tmp = args
        .voice_refs_dir
        .join(format!("{voice_id}.src.{src_suffix}"));
    let dst_audio = args.voice_refs_dir.join(format!("{voice_id}.wav"));
    let dst_sidecar = args.voice_refs_dir.join(format!("{voice_id}.json"));

    std::fs::write(&src_tmp, args.audio_bytes)?;

    let encode_result = run_ffmpeg(&src_tmp, &dst_audio);
    let _ = std::fs::remove_file(&src_tmp);
    encode_result.map_err(|e| {
        let _ = std::fs::remove_file(&dst_audio);
        e
    })?;

    let duration = match probe_duration(&dst_audio) {
        Ok(d) => d,
        Err(e) => {
            let _ = std::fs::remove_file(&dst_audio);
            return Err(e);
        }
    };
    if duration < args.min_duration_secs {
        let _ = std::fs::remove_file(&dst_audio);
        bail!(
            "audio is too short ({duration:.1}s < {:.1}s)",
            args.min_duration_secs
        );
    }
    if duration > args.max_duration_secs {
        let _ = std::fs::remove_file(&dst_audio);
        bail!(
            "audio is too long ({duration:.1}s > {:.1}s)",
            args.max_duration_secs
        );
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let info = VoiceRef {
        id: voice_id,
        name,
        ref_text: ref_text.to_string(),
        permission_confirmed: args.permission_confirmed,
        sample_rate: 24_000,
        duration_secs: round_to_3(duration),
        created_at: now,
        audio_path: dst_audio.to_string_lossy().to_string(),
        sidecar_path: dst_sidecar.to_string_lossy().to_string(),
    };
    let json = serde_json::to_string_pretty(&info)?;
    std::fs::write(&dst_sidecar, json)?;

    append_audit(args.voice_refs_dir, "upload", &info);
    Ok(info)
}

pub fn list_refs(voice_refs_dir: &Path) -> Vec<VoiceRef> {
    let mut refs: Vec<VoiceRef> = Vec::new();
    let Ok(rd) = std::fs::read_dir(voice_refs_dir) else {
        return refs;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        if p.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        match std::fs::read_to_string(&p) {
            Ok(text) => match serde_json::from_str::<VoiceRef>(&text) {
                Ok(info) => refs.push(info),
                Err(e) => eprintln!("[refs] WARN: skipping corrupt sidecar {p:?}: {e}"),
            },
            Err(e) => eprintln!("[refs] WARN: cannot read sidecar {p:?}: {e}"),
        }
    }
    refs.sort_by(|a, b| {
        b.created_at
            .partial_cmp(&a.created_at)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    refs
}

pub fn get_ref(voice_refs_dir: &Path, voice_id: &str) -> Result<Option<VoiceRef>> {
    let voice_id = sanitize_voice_id(voice_id)?;
    let sidecar = voice_refs_dir.join(format!("{voice_id}.json"));
    if !sidecar.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&sidecar)?;
    Ok(serde_json::from_str(&text).ok())
}

pub fn delete_ref(voice_refs_dir: &Path, voice_id: &str) -> Result<bool> {
    let voice_id = sanitize_voice_id(voice_id)?;
    let audio = voice_refs_dir.join(format!("{voice_id}.wav"));
    let sidecar = voice_refs_dir.join(format!("{voice_id}.json"));
    if !audio.exists() && !sidecar.exists() {
        return Ok(false);
    }
    let info = get_ref(voice_refs_dir, &voice_id).ok().flatten();
    let _ = std::fs::remove_file(&audio);
    let _ = std::fs::remove_file(&sidecar);
    if let Some(info) = info {
        append_audit(voice_refs_dir, "delete", &info);
    }
    Ok(true)
}

fn round_to_3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Re-encode `src` to 24 kHz mono PCM16 WAV at `dst` via the system
/// `ffmpeg` binary. Mirrors the Python helper.
pub(crate) fn run_ffmpeg(src: &Path, dst: &Path) -> Result<()> {
    let output = Command::new("ffmpeg")
        .args(["-y", "-i"])
        .arg(src)
        .args(["-ar", "24000", "-ac", "1", "-acodec", "pcm_s16le"])
        .arg(dst)
        .output()
        .map_err(|e| anyhow!("ffmpeg execution failed: {e} (apt install ffmpeg)"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(3).collect();
        let mut tail = tail;
        tail.reverse();
        bail!(
            "ffmpeg failed (rc={}): {}",
            output.status.code().unwrap_or(-1),
            tail.join(" / ")
        );
    }
    Ok(())
}

/// Return audio duration in seconds via the `hound` crate (assumes the
/// file is a real PCM WAV — i.e. post-`run_ffmpeg`).
pub(crate) fn probe_duration(path: &Path) -> Result<f64> {
    let reader =
        hound::WavReader::open(path).map_err(|e| anyhow!("opening WAV {path:?}: {e}"))?;
    let spec = reader.spec();
    if spec.sample_rate == 0 {
        bail!("WAV has zero sample rate at {path:?}");
    }
    Ok(reader.duration() as f64 / spec.sample_rate as f64)
}

/// Append-only audit log. Intentionally minimal — id, name, when,
/// permission flag, duration. NEVER includes the audio bytes or the
/// transcript. Format mirrors vllm-server's audit log line for parity.
fn append_audit(out_dir: &Path, action: &str, info: &VoiceRef) {
    let log = out_dir.join("_audit.log");
    let when = format_iso8601_utc(info.created_at);
    // `name` is sanitized (`[A-Za-z0-9 _-]{1,64}`) so the `:?` debug
    // format won't escape weird characters; we include the quotes to
    // match the Python output format.
    let line = format!(
        "{when}\t{action}\t{}\tname={:?}\tpermission={}\tduration_secs={}\n",
        info.id, info.name, info.permission_confirmed, info.duration_secs
    );
    use std::io::Write;
    if let Err(e) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log)
        .and_then(|mut f| f.write_all(line.as_bytes()))
    {
        eprintln!("[refs] WARN: audit log write failed: {e}");
    }
}

/// Format unix-epoch seconds as `YYYY-MM-DDTHH:MM:SS+00:00`. Crude
/// hand-rolled Gregorian calendar arithmetic — good enough for an
/// append-only audit log line, no external date crate dependency.
fn format_iso8601_utc(unix_secs: f64) -> String {
    let total_secs = unix_secs as i64;
    let total_days = total_secs.div_euclid(86_400);
    let day_secs = total_secs.rem_euclid(86_400);
    let hour = (day_secs / 3600) as u32;
    let minute = ((day_secs / 60) % 60) as u32;
    let second = (day_secs % 60) as u32;
    let (year, month, day) = days_since_epoch_to_ymd(total_days);
    format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}+00:00"
    )
}

fn is_leap_year(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_since_epoch_to_ymd(days: i64) -> (i64, u32, u32) {
    // Days are 0-indexed since 1970-01-01. Negative inputs aren't
    // expected (we get our timestamps from `SystemTime::now`).
    let mut year: i64 = 1970;
    let mut day = days;
    loop {
        let year_days = if is_leap_year(year) { 366 } else { 365 };
        if day < year_days {
            break;
        }
        day -= year_days;
        year += 1;
    }
    let leap = is_leap_year(year);
    let month_lens: [i64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month: u32 = 1;
    for &md in &month_lens {
        if day < md {
            break;
        }
        day -= md;
        month += 1;
    }
    (year, month, (day + 1) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmpdir(label: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "burn-tts-refs-test-{label}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn ffmpeg_available() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn write_silence_wav(path: &Path, secs: f64) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        let n = (secs * 24_000.0).round() as usize;
        for _ in 0..n {
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();
    }

    #[test]
    fn sanitize_name_accepts_valid_inputs() {
        assert_eq!(sanitize_name("Goran").unwrap(), "Goran");
        assert_eq!(sanitize_name("My Voice 1").unwrap(), "My Voice 1");
        assert_eq!(sanitize_name("Voice-test_2026").unwrap(), "Voice-test_2026");
        // Trims surrounding whitespace.
        assert_eq!(sanitize_name("  ok  ").unwrap(), "ok");
    }

    #[test]
    fn sanitize_name_rejects_bad_inputs() {
        assert!(sanitize_name("").is_err());
        assert!(sanitize_name("   ").is_err());
        // Disallowed punctuation:
        assert!(sanitize_name("Hello!").is_err());
        assert!(sanitize_name("../traversal").is_err());
        // Over-long:
        assert!(sanitize_name(&"a".repeat(65)).is_err());
    }

    #[test]
    fn sanitize_voice_id_rejects_path_traversal() {
        assert!(sanitize_voice_id("../etc/passwd").is_err());
        assert!(sanitize_voice_id("ABCDEF").is_err()); // upper-case
        assert!(sanitize_voice_id("short").is_err()); // <8
        assert!(sanitize_voice_id("0123456789abcdef0123456789abcdef").is_ok());
        assert!(sanitize_voice_id("0123abcd-4567-89ab-cdef-0123456789ab").is_ok());
    }

    #[test]
    fn list_refs_returns_empty_on_missing_dir() {
        let p = tmpdir("missing");
        std::fs::remove_dir_all(&p).ok();
        assert!(list_refs(&p).is_empty());
    }

    #[test]
    fn delete_ref_returns_false_for_unknown_id() {
        let p = tmpdir("missing-id");
        let r = delete_ref(&p, "deadbeef0123456789abcdef0123dead").unwrap();
        assert!(!r);
    }

    #[test]
    fn save_ref_round_trip_via_ffmpeg() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not installed");
            return;
        }
        let dir = tmpdir("roundtrip");

        // Create a 6-second silence WAV as the upload "audio_sample".
        let src = dir.join("input.wav");
        write_silence_wav(&src, 6.0);
        let bytes = std::fs::read(&src).unwrap();

        let info = save_ref(SaveRefArgs {
            audio_bytes: &bytes,
            audio_filename: "input.wav",
            name: "Test Voice",
            ref_text: "This is a transcript of the reference audio.",
            permission_confirmed: true,
            voice_refs_dir: &dir,
            max_audio_bytes: 5_000_000,
            min_duration_secs: 5.0,
            max_duration_secs: 30.0,
            require_permission: true,
        })
        .unwrap();

        assert_eq!(info.name, "Test Voice");
        assert_eq!(info.sample_rate, 24_000);
        assert!(info.duration_secs >= 5.5 && info.duration_secs <= 6.5);
        assert!(Path::new(&info.audio_path).is_file());
        assert!(Path::new(&info.sidecar_path).is_file());

        // List sees it.
        let listing = list_refs(&dir);
        assert_eq!(listing.len(), 1);
        assert_eq!(listing[0].id, info.id);

        // Get sees it.
        let got = get_ref(&dir, &info.id).unwrap().unwrap();
        assert_eq!(got.id, info.id);

        // Audit log got an "upload" line.
        let audit = std::fs::read_to_string(dir.join("_audit.log")).unwrap();
        assert!(audit.contains("upload"));
        assert!(audit.contains(&info.id));
        // CRITICAL: audit log MUST NOT leak the transcript.
        assert!(!audit.contains("transcript of the reference"));

        // Delete works + appends a "delete" audit line.
        let removed = delete_ref(&dir, &info.id).unwrap();
        assert!(removed);
        assert!(!Path::new(&info.audio_path).exists());
        assert!(!Path::new(&info.sidecar_path).exists());
        let audit = std::fs::read_to_string(dir.join("_audit.log")).unwrap();
        assert!(audit.contains("delete"));
    }

    #[test]
    fn save_ref_requires_permission_when_configured() {
        let dir = tmpdir("perm");
        let bytes = vec![0u8; 100];
        let err = save_ref(SaveRefArgs {
            audio_bytes: &bytes,
            audio_filename: "input.wav",
            name: "Voice",
            ref_text: "anything",
            permission_confirmed: false,
            voice_refs_dir: &dir,
            max_audio_bytes: 5_000_000,
            min_duration_secs: 5.0,
            max_duration_secs: 30.0,
            require_permission: true,
        })
        .unwrap_err();
        assert!(format!("{err:?}").contains("permission_confirmed"));
    }

    #[test]
    fn save_ref_rejects_empty_audio() {
        let dir = tmpdir("empty");
        let err = save_ref(SaveRefArgs {
            audio_bytes: &[],
            audio_filename: "input.wav",
            name: "Voice",
            ref_text: "anything",
            permission_confirmed: true,
            voice_refs_dir: &dir,
            max_audio_bytes: 5_000_000,
            min_duration_secs: 5.0,
            max_duration_secs: 30.0,
            require_permission: true,
        })
        .unwrap_err();
        assert!(format!("{err:?}").contains("empty"));
    }

    #[test]
    fn save_ref_rejects_bad_extension() {
        let dir = tmpdir("ext");
        let bytes = vec![0u8; 100];
        let err = save_ref(SaveRefArgs {
            audio_bytes: &bytes,
            audio_filename: "input.txt",
            name: "Voice",
            ref_text: "anything",
            permission_confirmed: true,
            voice_refs_dir: &dir,
            max_audio_bytes: 5_000_000,
            min_duration_secs: 5.0,
            max_duration_secs: 30.0,
            require_permission: true,
        })
        .unwrap_err();
        assert!(format!("{err:?}").contains("audio_filename"));
    }

    #[test]
    fn save_ref_rejects_empty_ref_text() {
        let dir = tmpdir("emptytext");
        let bytes = vec![0u8; 100];
        let err = save_ref(SaveRefArgs {
            audio_bytes: &bytes,
            audio_filename: "input.wav",
            name: "Voice",
            ref_text: "",
            permission_confirmed: true,
            voice_refs_dir: &dir,
            max_audio_bytes: 5_000_000,
            min_duration_secs: 5.0,
            max_duration_secs: 30.0,
            require_permission: true,
        })
        .unwrap_err();
        assert!(format!("{err:?}").contains("ref_text"));
    }

    #[test]
    fn iso8601_format_known_timestamps() {
        // 1970-01-01T00:00:00 epoch
        assert_eq!(format_iso8601_utc(0.0), "1970-01-01T00:00:00+00:00");
        // 2026-01-01T00:00:00 UTC = 1767225600
        assert_eq!(
            format_iso8601_utc(1767225600.0),
            "2026-01-01T00:00:00+00:00"
        );
    }

    #[test]
    fn audit_log_excludes_transcript_and_audio_bytes() {
        // Pure unit test that doesn't touch ffmpeg — verifies the audit
        // line shape directly. The full round-trip test above asserts
        // the same property end-to-end.
        let dir = tmpdir("audit-shape");
        let info = VoiceRef {
            id: "abcdef0123".to_string(),
            name: "Test".to_string(),
            ref_text: "secret transcript that must NOT leak".to_string(),
            permission_confirmed: true,
            sample_rate: 24_000,
            duration_secs: 6.0,
            created_at: 1767225600.0,
            audio_path: dir.join("abcdef0123.wav").to_string_lossy().to_string(),
            sidecar_path: dir.join("abcdef0123.json").to_string_lossy().to_string(),
        };
        append_audit(&dir, "upload", &info);
        let audit = std::fs::read_to_string(dir.join("_audit.log")).unwrap();
        assert!(audit.contains("upload"));
        assert!(audit.contains("abcdef0123"));
        assert!(audit.contains("name=\"Test\""));
        assert!(audit.contains("permission=true"));
        assert!(audit.contains("duration_secs=6"));
        // Must NOT leak the transcript or any audio bytes.
        assert!(!audit.contains("transcript"));
        assert!(!audit.contains("must NOT leak"));
    }
}
