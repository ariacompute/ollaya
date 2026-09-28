//! Public weight hubs: Hugging Face vs ModelScope.
//!
//! Preference follows `OLLAYA_HUB` (`huggingface` | `modelscope` | `auto`).
//! `auto` picks ModelScope when the process locale or `LANG` looks Chinese
//! (aligned with Aria Engine's `.cn` → ModelScope rule).

use std::sync::OnceLock;

/// Where foreign weight layers are fetched from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicHub {
    HuggingFace,
    ModelScope,
}

impl PublicHub {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HuggingFace => "huggingface",
            Self::ModelScope => "modelscope",
        }
    }

    /// True when `url` is served by this hub.
    pub fn matches_url(self, url: &str) -> bool {
        match self {
            Self::HuggingFace => url.contains("huggingface.co/") || url.contains("hf-mirror.com/"),
            Self::ModelScope => url.contains("modelscope.cn/") || url.contains("modelscope.com/"),
        }
    }
}

/// Preferred hub for this process (cached after first read).
pub fn preferred_hub() -> PublicHub {
    static HUB: OnceLock<PublicHub> = OnceLock::new();
    *HUB.get_or_init(detect_hub)
}

fn detect_hub() -> PublicHub {
    match std::env::var("OLLAYA_HUB")
        .unwrap_or_else(|_| "auto".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "huggingface" | "hf" => PublicHub::HuggingFace,
        "modelscope" | "ms" => PublicHub::ModelScope,
        _ => {
            if locale_prefers_modelscope() {
                PublicHub::ModelScope
            } else {
                PublicHub::HuggingFace
            }
        }
    }
}

fn locale_prefers_modelscope() -> bool {
    for key in ["OLLAYA_LANG", "LC_ALL", "LC_MESSAGES", "LANG"] {
        if let Ok(v) = std::env::var(key) {
            let lower = v.to_ascii_lowercase();
            if lower.starts_with("zh") || lower.contains(".cn") {
                return true;
            }
        }
    }
    false
}

/// Pick a blob URL from a descriptor's `urls` list for the preferred hub.
/// Prefers a URL matching the hub; otherwise the first URL; empty → `None`.
pub fn select_url(urls: &[String], hub: PublicHub) -> Option<&str> {
    urls.iter()
        .find(|u| hub.matches_url(u))
        .or_else(|| urls.first())
        .map(|s| s.as_str())
}

/// Alternate URL from the other hub (for download fallback).
pub fn fallback_url<'a>(urls: &'a [String], hub: PublicHub, primary: &str) -> Option<&'a str> {
    let other = match hub {
        PublicHub::HuggingFace => PublicHub::ModelScope,
        PublicHub::ModelScope => PublicHub::HuggingFace,
    };
    urls.iter()
        .find(|u| other.matches_url(u) && u.as_str() != primary)
        .map(|s| s.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_prefers_matching_hub() {
        let urls = vec![
            "https://huggingface.co/ariacompute/afm-de/resolve/abc/model.safetensors".into(),
            "https://www.modelscope.cn/models/AriaCompute/afm-de/resolve/master/model.safetensors"
                .into(),
        ];
        assert!(
            select_url(&urls, PublicHub::HuggingFace)
                .unwrap()
                .contains("huggingface.co")
        );
        assert!(
            select_url(&urls, PublicHub::ModelScope)
                .unwrap()
                .contains("modelscope.cn")
        );
    }

    #[test]
    fn select_falls_back_to_first() {
        let urls = vec!["https://ollaya.dev/blobs/sha256-dead".into()];
        assert_eq!(
            select_url(&urls, PublicHub::ModelScope),
            Some("https://ollaya.dev/blobs/sha256-dead")
        );
    }
}
