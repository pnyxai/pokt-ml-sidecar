use base64::{engine::general_purpose::STANDARD, Engine as _};
use lofty::file::{AudioFile, FileType, TaggedFileExt};
use lofty::probe::Probe;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::io::Cursor;
use webm_iterable::matroska_spec::{Master, MatroskaSpec};
use webm_iterable::WebmIterator;

const DEFAULT_MAX_ITEM_BYTES: usize = 5 * 1024 * 1024;
const DEFAULT_MAX_IMAGE_DIMENSION: u32 = 8192;
const DEFAULT_MAX_AUDIO_DURATION_SECONDS: f64 = 600.0;
const DEFAULT_MAX_VIDEO_DIMENSION: u32 = 8192;
const DEFAULT_MAX_VIDEO_DURATION_SECONDS: f64 = 600.0;
const WEBM_DEFAULT_TIMESTAMP_SCALE: u64 = 1_000_000;

/// Per-modality item count or byte limits.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default, PartialEq, Eq)]
pub struct ModalityLimits {
    #[serde(default)]
    pub image: usize,
    #[serde(default)]
    pub audio: usize,
    #[serde(default)]
    pub video: usize,
}

/// Media validation policy, expected to mirror the engine's
/// `--limit-mm-per-prompt` configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MediaConfig {
    /// Whether `video_url` content parts are accepted.
    #[serde(default)]
    pub allow_video: bool,
    /// Maximum number of media items per request for each modality.
    #[serde(default)]
    pub limit_mm_per_prompt: ModalityLimits,
    /// Maximum decoded bytes per media item for each modality.
    #[serde(default = "default_max_item_bytes")]
    pub max_item_bytes: ModalityLimits,
    /// Maximum accepted image width in pixels.
    #[serde(default = "default_max_image_dimension")]
    pub max_image_width: u32,
    /// Maximum accepted image height in pixels.
    #[serde(default = "default_max_image_dimension")]
    pub max_image_height: u32,
    /// Maximum accepted audio duration in seconds.
    #[serde(default = "default_max_audio_duration")]
    pub max_audio_duration_seconds: f64,
    /// Maximum accepted video width in pixels.
    #[serde(default = "default_max_video_dimension")]
    pub max_video_width: u32,
    /// Maximum accepted video height in pixels.
    #[serde(default = "default_max_video_dimension")]
    pub max_video_height: u32,
    /// Maximum accepted video duration in seconds.
    #[serde(default = "default_max_video_duration")]
    pub max_video_duration_seconds: f64,
}

impl Default for MediaConfig {
    fn default() -> Self {
        Self {
            allow_video: false,
            limit_mm_per_prompt: ModalityLimits::default(),
            max_item_bytes: default_max_item_bytes(),
            max_image_width: default_max_image_dimension(),
            max_image_height: default_max_image_dimension(),
            max_audio_duration_seconds: default_max_audio_duration(),
            max_video_width: default_max_video_dimension(),
            max_video_height: default_max_video_dimension(),
            max_video_duration_seconds: default_max_video_duration(),
        }
    }
}

fn default_max_item_bytes() -> ModalityLimits {
    ModalityLimits {
        image: DEFAULT_MAX_ITEM_BYTES,
        audio: DEFAULT_MAX_ITEM_BYTES,
        video: DEFAULT_MAX_ITEM_BYTES,
    }
}

fn default_max_image_dimension() -> u32 {
    DEFAULT_MAX_IMAGE_DIMENSION
}

fn default_max_audio_duration() -> f64 {
    DEFAULT_MAX_AUDIO_DURATION_SECONDS
}

fn default_max_video_dimension() -> u32 {
    DEFAULT_MAX_VIDEO_DIMENSION
}

fn default_max_video_duration() -> f64 {
    DEFAULT_MAX_VIDEO_DURATION_SECONDS
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MediaKind {
    Png,
    Jpeg,
    Webp,
    Wav,
    Flac,
    Mp3,
    Mp4,
    Webm,
}

impl MediaKind {
    fn image_mime(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Webp => "image/webp",
            _ => unreachable!("not an image kind"),
        }
    }

    fn audio_format(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            _ => unreachable!("not an audio kind"),
        }
    }

    fn video_mime(self) -> &'static str {
        match self {
            Self::Mp4 => "video/mp4",
            Self::Webm => "video/webm",
            _ => unreachable!("not a video kind"),
        }
    }
}

#[derive(Debug, Default)]
struct MediaCounts {
    image: usize,
    audio: usize,
    video: usize,
}

/// Validates all media content parts and strips client-supplied UUIDs.
///
/// Media is only accepted inline as base64 data URIs (or OpenAI's
/// `input_audio.data` form). Declared MIME types are ignored; the decoded
/// bytes determine the actual format and are rewritten to a canonical data URI
/// before forwarding.
pub fn validate_and_sanitize(
    obj: &mut Map<String, Value>,
    config: &MediaConfig,
) -> Result<(), String> {
    let Some(messages) = obj.get_mut("messages") else {
        return Ok(());
    };
    let Some(messages) = messages.as_array_mut() else {
        return Ok(());
    };

    let mut counts = MediaCounts::default();

    for message in messages {
        let Some(message) = message.as_object_mut() else {
            continue;
        };
        let Some(content) = message.get_mut("content") else {
            continue;
        };
        let Some(parts) = content.as_array_mut() else {
            continue;
        };

        for part in parts {
            let Some(part) = part.as_object_mut() else {
                return Err("content parts must be JSON objects".to_string());
            };

            part.remove("uuid");
            remove_nested_uuid(part, "image_url");
            remove_nested_uuid(part, "video_url");
            remove_nested_uuid(part, "input_audio");

            let part_type = part
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| "content part type must be a string".to_string())?;

            match part_type {
                "text" => {}
                "image_url" => {
                    counts.image = counts.image.saturating_add(1);
                    check_count("image", counts.image, config.limit_mm_per_prompt.image)?;

                    let image = part
                        .get_mut("image_url")
                        .and_then(Value::as_object_mut)
                        .ok_or_else(|| "image_url must be an object".to_string())?;
                    let url = image
                        .get("url")
                        .and_then(Value::as_str)
                        .ok_or_else(|| "image_url.url must be a string".to_string())?;

                    let bytes = decode_data_uri(url, config.max_item_bytes.image)?;
                    let kind = sniff_media(&bytes)
                        .filter(|kind| {
                            matches!(kind, MediaKind::Png | MediaKind::Jpeg | MediaKind::Webp)
                        })
                        .ok_or_else(|| "unsupported image format".to_string())?;
                    check_image_dimensions(&bytes, config)?;
                    image.insert(
                        "url".to_string(),
                        Value::String(encode_data_uri(kind.image_mime(), &bytes)),
                    );
                }
                "input_audio" => {
                    counts.audio = counts.audio.saturating_add(1);
                    check_count("audio", counts.audio, config.limit_mm_per_prompt.audio)?;

                    let audio = part
                        .get_mut("input_audio")
                        .and_then(Value::as_object_mut)
                        .ok_or_else(|| "input_audio must be an object".to_string())?;

                    let bytes = if let Some(data) = audio.get("data").and_then(Value::as_str) {
                        decode_base64(data, config.max_item_bytes.audio)?
                    } else if let Some(url) = audio.get("url").and_then(Value::as_str) {
                        decode_data_uri(url, config.max_item_bytes.audio)?
                    } else {
                        return Err(
                            "input_audio must contain base64 data or a data URI".to_string()
                        );
                    };

                    let kind = validate_audio_bytes(&bytes, config)?;
                    audio.remove("url");
                    audio.insert("data".to_string(), Value::String(STANDARD.encode(&bytes)));
                    audio.insert(
                        "format".to_string(),
                        Value::String(kind.audio_format().to_string()),
                    );
                }
                "video_url" => {
                    if !config.allow_video {
                        return Err("video_url is disabled".to_string());
                    }
                    counts.video = counts.video.saturating_add(1);
                    check_count("video", counts.video, config.limit_mm_per_prompt.video)?;

                    let video = part
                        .get_mut("video_url")
                        .and_then(Value::as_object_mut)
                        .ok_or_else(|| "video_url must be an object".to_string())?;
                    let url = video
                        .get("url")
                        .and_then(Value::as_str)
                        .ok_or_else(|| "video_url.url must be a string".to_string())?;

                    let bytes = decode_data_uri(url, config.max_item_bytes.video)?;
                    let kind = validate_video_bytes(&bytes, config)?;
                    video.insert(
                        "url".to_string(),
                        Value::String(encode_data_uri(kind.video_mime(), &bytes)),
                    );
                }
                other => {
                    return Err(format!("unsupported content part type: {}", other));
                }
            }
        }
    }

    Ok(())
}

fn remove_nested_uuid(part: &mut Map<String, Value>, field: &str) {
    if let Some(nested) = part.get_mut(field).and_then(Value::as_object_mut) {
        nested.remove("uuid");
    }
}

fn check_count(modality: &str, count: usize, limit: usize) -> Result<(), String> {
    if count > limit {
        return Err(format!(
            "too many {} parts: {} > limit {}",
            modality, count, limit
        ));
    }
    Ok(())
}

/// Decodes a strict base64 data URI and enforces the decoded byte limit.
fn decode_data_uri(uri: &str, max_bytes: usize) -> Result<Vec<u8>, String> {
    if uri.len() < 5 || !uri.as_bytes()[..5].eq_ignore_ascii_case(b"data:") {
        return Err("media must be an inline data: URI".to_string());
    }

    let (header, payload) = uri[5..]
        .split_once(',')
        .ok_or_else(|| "malformed data URI".to_string())?;

    let mut header_parts = header.split(';');
    let mime = header_parts.next().unwrap_or_default();
    if mime.is_empty() {
        return Err("data URI must include a media type".to_string());
    }

    let mut is_base64 = false;
    for parameter in header_parts {
        if parameter.eq_ignore_ascii_case("base64") {
            is_base64 = true;
        } else {
            return Err("only base64 data URIs are accepted".to_string());
        }
    }
    if !is_base64 {
        return Err("only base64 data URIs are accepted".to_string());
    }

    decode_base64(payload, max_bytes)
}

/// Decodes a base64 payload and enforces the decoded byte limit.
fn decode_base64(payload: &str, max_bytes: usize) -> Result<Vec<u8>, String> {
    if payload.is_empty() {
        return Err("media payload must not be empty".to_string());
    }

    let max_encoded_len = max_bytes
        .saturating_add(2)
        .saturating_div(3)
        .saturating_mul(4)
        .saturating_add(4);
    if payload.len() > max_encoded_len {
        return Err(format!("media item exceeds the {} byte limit", max_bytes));
    }

    let bytes = STANDARD
        .decode(payload)
        .map_err(|_| "invalid base64 payload".to_string())?;

    if bytes.is_empty() {
        return Err("media payload must not be empty".to_string());
    }
    if bytes.len() > max_bytes {
        return Err(format!("media item exceeds the {} byte limit", max_bytes));
    }

    Ok(bytes)
}

fn encode_data_uri(mime: &str, bytes: &[u8]) -> String {
    format!("data:{};base64,{}", mime, STANDARD.encode(bytes))
}

/// Detects the actual media format from magic bytes.
fn sniff_media(bytes: &[u8]) -> Option<MediaKind> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some(MediaKind::Png);
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(MediaKind::Jpeg);
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some(MediaKind::Webp);
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return Some(MediaKind::Wav);
    }
    if bytes.starts_with(b"fLaC") {
        return Some(MediaKind::Flac);
    }
    if bytes.starts_with(b"ID3") {
        return Some(MediaKind::Mp3);
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0 {
        return Some(MediaKind::Mp3);
    }
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        return Some(MediaKind::Mp4);
    }
    if bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return Some(MediaKind::Webm);
    }

    None
}

fn check_image_dimensions(bytes: &[u8], config: &MediaConfig) -> Result<(), String> {
    let size =
        imagesize::blob_size(bytes).map_err(|_| "unable to read image dimensions".to_string())?;

    if size.width as u64 > config.max_image_width as u64
        || size.height as u64 > config.max_image_height as u64
    {
        return Err(format!(
            "image dimensions {}x{} exceed the {}x{} limit",
            size.width, size.height, config.max_image_width, config.max_image_height
        ));
    }

    Ok(())
}

fn validate_audio_bytes(bytes: &[u8], config: &MediaConfig) -> Result<MediaKind, String> {
    let sniffed = sniff_media(bytes).ok_or_else(|| "unsupported audio format".to_string())?;
    if !matches!(sniffed, MediaKind::Wav | MediaKind::Flac | MediaKind::Mp3) {
        return Err("unsupported audio format".to_string());
    }

    let tagged = Probe::new(Cursor::new(bytes))
        .guess_file_type()
        .map_err(|_| "unable to detect audio format".to_string())?
        .read()
        .map_err(|_| "unable to parse audio metadata".to_string())?;

    let kind = match tagged.file_type() {
        FileType::Wav => MediaKind::Wav,
        FileType::Flac => MediaKind::Flac,
        FileType::Mpeg => MediaKind::Mp3,
        _ => return Err("unsupported audio format".to_string()),
    };

    let duration = tagged.properties().duration();
    if duration.is_zero() {
        return Err("audio duration is not available".to_string());
    }
    if duration.as_secs_f64() > config.max_audio_duration_seconds {
        return Err(format!(
            "audio duration {:.3}s exceeds the {}s limit",
            duration.as_secs_f64(),
            config.max_audio_duration_seconds
        ));
    }

    Ok(kind)
}

fn validate_video_bytes(bytes: &[u8], config: &MediaConfig) -> Result<MediaKind, String> {
    let kind = sniff_media(bytes)
        .filter(|kind| matches!(kind, MediaKind::Mp4 | MediaKind::Webm))
        .ok_or_else(|| "unsupported video format".to_string())?;

    let (width, height, duration) = match kind {
        MediaKind::Mp4 => parse_mp4_metadata(bytes)?,
        MediaKind::Webm => parse_webm_metadata(bytes)?,
        _ => unreachable!("sniff_media only returns video kinds here"),
    };

    if width > config.max_video_width as u64 || height > config.max_video_height as u64 {
        return Err(format!(
            "video dimensions {}x{} exceed the {}x{} limit",
            width, height, config.max_video_width, config.max_video_height
        ));
    }
    if duration > config.max_video_duration_seconds {
        return Err(format!(
            "video duration {:.3}s exceeds the {}s limit",
            duration, config.max_video_duration_seconds
        ));
    }

    Ok(kind)
}

/// Returns the largest video-track dimensions and duration in an MP4 container.
fn parse_mp4_metadata(bytes: &[u8]) -> Result<(u64, u64, f64), String> {
    let mut cursor = Cursor::new(bytes);
    let context =
        mp4parse::read_mp4(&mut cursor).map_err(|_| "unable to parse MP4 metadata".to_string())?;

    let mut video: Option<(u64, u64, f64)> = None;

    for track in context.tracks.iter() {
        if track.track_type != mp4parse::TrackType::Video {
            continue;
        }

        let duration = match (track.duration, track.timescale) {
            (Some(duration), Some(timescale)) if duration.0 != u64::MAX && timescale.0 > 0 => {
                duration.0 as f64 / timescale.0 as f64
            }
            _ => return Err("video duration is not available".to_string()),
        };
        if duration <= 0.0 {
            return Err("video duration is not available".to_string());
        }

        let (width, height) = mp4_video_dimensions(track)
            .ok_or_else(|| "video dimensions are not available".to_string())?;

        video = Some(match video {
            Some((w, h, d)) => (w.max(width), h.max(height), d.max(duration)),
            None => (width, height, duration),
        });
    }

    video.ok_or_else(|| "video contains no video track".to_string())
}

fn mp4_video_dimensions(track: &mp4parse::Track) -> Option<(u64, u64)> {
    let mut dimensions: Option<(u64, u64)> = None;

    if let Some(stsd) = &track.stsd {
        for entry in stsd.descriptions.iter() {
            if let mp4parse::SampleEntry::Video(sample) = entry {
                if sample.width > 0 && sample.height > 0 {
                    let (width, height) = (sample.width as u64, sample.height as u64);
                    dimensions = Some(match dimensions {
                        Some((w, h)) => (w.max(width), h.max(height)),
                        None => (width, height),
                    });
                }
            }
        }
    }

    if dimensions.is_some() {
        return dimensions;
    }

    track
        .tkhd
        .as_ref()
        .map(|tkhd| ((tkhd.width >> 16) as u64, (tkhd.height >> 16) as u64))
        .filter(|(width, height)| *width > 0 && *height > 0)
}

/// Returns the largest video-track dimensions and duration in a WebM container.
fn parse_webm_metadata(bytes: &[u8]) -> Result<(u64, u64, f64), String> {
    let mut cursor = Cursor::new(bytes);
    let mut duration: Option<f64> = None;
    let mut timestamp_scale = WEBM_DEFAULT_TIMESTAMP_SCALE;
    let mut in_video = false;
    let mut width: Option<u64> = None;
    let mut height: Option<u64> = None;

    for tag in WebmIterator::new(&mut cursor, &[]) {
        let tag = tag.map_err(|_| "unable to parse WebM metadata".to_string())?;
        match tag {
            MatroskaSpec::TimestampScale(scale) => timestamp_scale = scale,
            MatroskaSpec::Duration(value) => duration = Some(value),
            MatroskaSpec::Video(Master::Start) => in_video = true,
            MatroskaSpec::Video(Master::End) => in_video = false,
            MatroskaSpec::PixelWidth(value) if in_video => {
                width = Some(width.map_or(value, |current| current.max(value)));
            }
            MatroskaSpec::PixelHeight(value) if in_video => {
                height = Some(height.map_or(value, |current| current.max(value)));
            }
            _ => {}
        }
    }

    let duration = duration.ok_or_else(|| "video duration is not available".to_string())?;
    let duration_seconds = duration * timestamp_scale as f64 / 1_000_000_000.0;
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Err("video duration is not available".to_string());
    }

    let width = width.ok_or_else(|| "video dimensions are not available".to_string())?;
    let height = height.ok_or_else(|| "video dimensions are not available".to_string())?;

    Ok((width, height, duration_seconds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config(image: usize, audio: usize) -> MediaConfig {
        MediaConfig {
            allow_video: false,
            limit_mm_per_prompt: ModalityLimits {
                image,
                audio,
                video: 0,
            },
            ..MediaConfig::default()
        }
    }

    fn png_1x1() -> Vec<u8> {
        STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
            .unwrap()
    }

    #[test]
    fn rejects_non_data_and_non_base64_uris() {
        for uri in [
            "http://example.com/a.png",
            "https://example.com/a.png",
            "file:///etc/passwd",
            "ftp://example.com/a.png",
            "/etc/passwd",
            "dätä:image/png;base64,AAAA",
            "data:image/png,plain",
            "data:image/png;base64,",
            "data:image/png;base64,not base64!",
        ] {
            assert!(decode_data_uri(uri, 1024).is_err(), "{}", uri);
        }
    }

    #[test]
    fn rejects_dangerous_magic_even_when_declared_as_an_image() {
        for bytes in [
            b"%PDF-1.7\n".as_slice(),
            b"%!PS-Adobe-3.0\n".as_slice(),
            b"\xC5\xD0\xD3\xC6".as_slice(),
            b"II*\x00".as_slice(),
            b"MM\x00*".as_slice(),
        ] {
            let uri = encode_data_uri("image/png", bytes);
            let decoded = decode_data_uri(&uri, 1024).unwrap();
            assert!(sniff_media(&decoded)
                .filter(|kind| matches!(kind, MediaKind::Png | MediaKind::Jpeg | MediaKind::Webp))
                .is_none());
        }
    }

    #[test]
    fn strips_uuid_from_media_parts() {
        let bytes = png_1x1();
        let mut payload = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "uuid": "forged"},
                    {
                        "type": "image_url",
                        "uuid": "forged",
                        "image_url": {
                            "url": encode_data_uri("image/png", &bytes),
                            "uuid": "forged-nested"
                        }
                    }
                ]
            }]
        });
        let obj = payload.as_object_mut().unwrap();

        validate_and_sanitize(obj, &config(1, 0)).unwrap();

        let parts = obj["messages"][0]["content"].as_array().unwrap();
        assert!(parts[0].get("uuid").is_none());
        assert!(parts[1].get("uuid").is_none());
        assert!(parts[1]["image_url"].get("uuid").is_none());
    }

    #[test]
    fn enforces_item_counts_and_byte_limits() {
        let bytes = png_1x1();
        let part = json!({
            "type": "image_url",
            "image_url": {"url": encode_data_uri("image/png", &bytes)}
        });
        let mut payload = json!({
            "messages": [{"role": "user", "content": [part.clone(), part]}]
        });
        let obj = payload.as_object_mut().unwrap();

        assert!(validate_and_sanitize(obj, &config(1, 0)).is_err());

        let mut small_limit = config(2, 0);
        small_limit.max_item_bytes.image = 8;
        let mut payload = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": encode_data_uri("image/png", &bytes)}
                }]
            }]
        });
        let obj = payload.as_object_mut().unwrap();
        assert!(validate_and_sanitize(obj, &small_limit).is_err());
    }

    #[test]
    fn enforces_image_dimensions() {
        let bytes = png_1x1();
        let mut payload = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": encode_data_uri("image/png", &bytes)}
                }]
            }]
        });
        let obj = payload.as_object_mut().unwrap();
        let mut config = config(1, 0);
        config.max_image_width = 0;
        assert!(validate_and_sanitize(obj, &config).is_err());
    }

    #[test]
    fn rejects_video_when_disabled() {
        let mut payload = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "video_url",
                    "video_url": {"url": "data:video/mp4;base64,AAAA"}
                }]
            }]
        });
        let obj = payload.as_object_mut().unwrap();
        assert!(validate_and_sanitize(obj, &config(0, 0)).is_err());
    }

    #[test]
    fn accepts_supported_audio_formats_and_normalizes_shape() {
        let cases: [(&[u8], &str); 3] = [
            (include_bytes!("../tests/fixtures/tiny.wav"), "wav"),
            (include_bytes!("../tests/fixtures/tiny.flac"), "flac"),
            (include_bytes!("../tests/fixtures/tiny.mp3"), "mp3"),
        ];

        for (bytes, expected_format) in cases {
            for uri_style in [false, true] {
                let audio = if uri_style {
                    json!({
                        "url": encode_data_uri("application/octet-stream", bytes)
                    })
                } else {
                    json!({
                        "data": STANDARD.encode(bytes),
                        "format": "not-trusted"
                    })
                };
                let mut payload = json!({
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "input_audio",
                            "input_audio": audio
                        }]
                    }]
                });
                let obj = payload.as_object_mut().unwrap();

                validate_and_sanitize(obj, &config(0, 1)).unwrap();

                let audio = &obj["messages"][0]["content"][0]["input_audio"];
                assert_eq!(audio["format"], expected_format);
                assert!(audio.get("data").is_some());
                assert!(audio.get("url").is_none());
                assert_eq!(
                    STANDARD.decode(audio["data"].as_str().unwrap()).unwrap(),
                    bytes
                );
            }
        }
    }

    #[test]
    fn enforces_audio_duration_limit() {
        let bytes = include_bytes!("../tests/fixtures/tiny.wav");
        let mut payload = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": STANDARD.encode(bytes), "format": "wav"}
                }]
            }]
        });
        let obj = payload.as_object_mut().unwrap();
        let mut config = config(0, 1);
        config.max_audio_duration_seconds = 0.01;

        assert!(validate_and_sanitize(obj, &config).is_err());
    }

    fn video_config() -> MediaConfig {
        let mut config = config(0, 0);
        config.allow_video = true;
        config.limit_mm_per_prompt.video = 1;
        config
    }

    fn video_payload(bytes: &[u8]) -> Value {
        json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "video_url",
                    "video_url": {"url": encode_data_uri("application/octet-stream", bytes)}
                }]
            }]
        })
    }

    const TINY_MP4: &[u8] = include_bytes!("../tests/fixtures/tiny.mp4");
    const TINY_WEBM: &[u8] = include_bytes!("../tests/fixtures/tiny.webm");

    #[test]
    fn accepts_video_when_enabled_and_canonicalizes_mime() {
        for (bytes, mime) in [(TINY_MP4, "video/mp4"), (TINY_WEBM, "video/webm")] {
            let mut payload = video_payload(bytes);
            let obj = payload.as_object_mut().unwrap();

            validate_and_sanitize(obj, &video_config()).unwrap();

            let url = obj["messages"][0]["content"][0]["video_url"]["url"]
                .as_str()
                .unwrap();
            assert!(url.starts_with(&format!("data:{};base64,", mime)));
        }
    }

    #[test]
    fn enforces_video_dimensions() {
        for bytes in [TINY_MP4, TINY_WEBM] {
            let mut payload = video_payload(bytes);
            let obj = payload.as_object_mut().unwrap();
            let mut config = video_config();
            config.max_video_width = 100;
            assert!(validate_and_sanitize(obj, &config).is_err());

            let mut payload = video_payload(bytes);
            let obj = payload.as_object_mut().unwrap();
            let mut config = video_config();
            config.max_video_height = 100;
            assert!(validate_and_sanitize(obj, &config).is_err());
        }
    }

    #[test]
    fn enforces_video_duration_limit() {
        for bytes in [TINY_MP4, TINY_WEBM] {
            let mut payload = video_payload(bytes);
            let obj = payload.as_object_mut().unwrap();
            let mut config = video_config();
            config.max_video_duration_seconds = 0.5;
            assert!(validate_and_sanitize(obj, &config).is_err());
        }
    }

    #[test]
    fn rejects_video_without_metadata() {
        for bytes in [
            b"\x00\x00\x00\x18ftypisom".as_slice(),
            b"\x1A\x45\xDF\xA3".as_slice(),
        ] {
            let mut payload = video_payload(bytes);
            let obj = payload.as_object_mut().unwrap();
            assert!(validate_and_sanitize(obj, &video_config()).is_err());
        }
    }

    #[test]
    fn rejects_truncated_video() {
        for bytes in [&TINY_MP4[..32], &TINY_WEBM[..32]] {
            let mut payload = video_payload(bytes);
            let obj = payload.as_object_mut().unwrap();
            assert!(validate_and_sanitize(obj, &video_config()).is_err());
        }
    }

    #[test]
    fn rejects_unknown_content_part_types() {
        let mut payload = json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "file", "file": {"file_data": "data:"}}]
            }]
        });
        let obj = payload.as_object_mut().unwrap();
        assert!(validate_and_sanitize(obj, &config(1, 1)).is_err());
    }
}
