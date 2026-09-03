//! id2label from config.json + transformers `aggregation_strategy="simple"` grouping.
use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::Error;
use crate::types::{EntityType, Span};

#[derive(Deserialize)]
struct Config { id2label: BTreeMap<String, String> }

/// Per-label (bi, type): `B-X` → (true, Some(X)), `I-X` → (false, Some(X)), `O`/unknown → (false, None) with the raw tag kept.
#[derive(Clone, Debug)]
pub struct Label { pub is_begin: bool, pub tag: String, pub entity: Option<EntityType> }

pub struct Labels(pub Vec<Label>);

impl Labels {
    pub fn load(config_json: &Path) -> Result<Labels, Error> {
        let cfg: Config = serde_json::from_slice(&std::fs::read(config_json)?)?;
        let n = cfg.id2label.len();
        let mut out = vec![None; n];
        for (k, v) in cfg.id2label {
            let i: usize = k.parse().map_err(|_| Error::Bundle(format!("id2label key {k:?}")))?;
            if i >= n { return Err(Error::Bundle(format!("id2label id {i} >= {n}"))); }
            out[i] = Some(parse_label(&v));
        }
        let labels = out.into_iter().enumerate()
            .map(|(i, l)| l.ok_or_else(|| Error::Bundle(format!("id2label missing id {i}"))))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Labels(labels))
    }
}

/// transformers `get_tag`: `B-`/`I-` prefix split; anything else is treated as `I-<name>`.
pub fn parse_label(name: &str) -> Label {
    let (is_begin, tag) = match name.strip_prefix("B-") {
        Some(t) => (true, t),
        None => (false, name.strip_prefix("I-").unwrap_or(name)),
    };
    Label { is_begin, tag: tag.to_string(), entity: EntityType::parse(tag) }
}

/// One non-special token: argmax label id, its score, char offsets.
pub struct TokenPred { pub label: usize, pub score: f32, pub start: usize, pub end: usize }

/// `group_entities` for the simple strategy: consecutive tokens join while the tag matches and the
/// token is not `B-`; group start/end = first/last token offsets, score = mean. `O` groups are
/// dropped (`ignore_labels=["O"]`), as are tags without a known EntityType.
pub fn group_simple(labels: &Labels, toks: &[TokenPred]) -> Vec<Span> {
    let mut out = Vec::new();
    let mut group: Vec<&TokenPred> = Vec::new();
    let flush = |group: &[&TokenPred], out: &mut Vec<Span>| {
        if let Some(first) = group.first() {
            let lab = &labels.0[first.label];
            if let Some(ent) = lab.entity {
                let last = group[group.len() - 1];
                let score = group.iter().map(|t| t.score).sum::<f32>() / group.len() as f32;
                // zero-width groups (all tokens with empty offsets) cannot exist as reference spans — the Python
                // `Span` constructor raises on start >= end — so they are dropped here rather than propagated
                if last.end > first.start { out.push(Span::new(first.start, last.end, ent, score)); }
            }
        }
    };
    for t in toks {
        if let Some(prev) = group.last() {
            let cur = &labels.0[t.label];
            let last = &labels.0[prev.label];
            if cur.tag == last.tag && !cur.is_begin {
                group.push(t);
                continue;
            }
            flush(&group, &mut out);
            group.clear();
        }
        group.push(t);
    }
    flush(&group, &mut out);
    out
}
