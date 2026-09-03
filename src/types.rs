//! Entity types and spans. 13 model (NER) types plus `Num`, a backstop-only catch-all for long digit runs
//! that the model did not label (`backstop = true`); `Num` never comes out of the model itself.
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EntityType {
    Rrn, Brn, Frn, Passport, DriverLicense, Card, Phone, Account, Email, Url, Code, Name, Address,
    Num,
}

impl EntityType {
    /// The 13 types the model predicts.
    pub const TRAINED: [EntityType; 13] = [
        EntityType::Rrn, EntityType::Brn, EntityType::Frn, EntityType::Passport, EntityType::DriverLicense,
        EntityType::Card, EntityType::Phone, EntityType::Account, EntityType::Email, EntityType::Url,
        EntityType::Code, EntityType::Name, EntityType::Address,
    ];
    /// Every type a span can carry: the model types plus the backstop-only `Num`.
    pub const ALL: [EntityType; 14] = [
        EntityType::Rrn, EntityType::Brn, EntityType::Frn, EntityType::Passport, EntityType::DriverLicense,
        EntityType::Card, EntityType::Phone, EntityType::Account, EntityType::Email, EntityType::Url,
        EntityType::Code, EntityType::Name, EntityType::Address, EntityType::Num,
    ];

    /// Label name as used in the model config (`B-RRN` → `RRN`).
    pub fn as_str(self) -> &'static str {
        match self {
            EntityType::Rrn => "RRN", EntityType::Brn => "BRN", EntityType::Frn => "FRN",
            EntityType::Passport => "PASSPORT", EntityType::DriverLicense => "DRIVER_LICENSE",
            EntityType::Card => "CARD", EntityType::Phone => "PHONE", EntityType::Account => "ACCOUNT",
            EntityType::Email => "EMAIL", EntityType::Url => "URL", EntityType::Code => "CODE",
            EntityType::Name => "NAME", EntityType::Address => "ADDRESS", EntityType::Num => "NUM",
        }
    }

    pub fn parse(s: &str) -> Option<EntityType> {
        EntityType::ALL.iter().copied().find(|t| t.as_str() == s)
    }
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.as_str()) }
}

/// Character-offset span (`start..end` in Unicode scalar values, like Python `str` indexing).
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub entity: EntityType,
    pub score: f32,
}

impl Span {
    pub fn new(start: usize, end: usize, entity: EntityType, score: f32) -> Span {
        debug_assert!(start < end, "empty span {start}..{end}");
        Span { start, end, entity, score }
    }
}

/// Slice `text` by char offsets (mirrors Python `text[start:end]`).
pub fn char_slice(chars: &[char], start: usize, end: usize) -> String {
    chars[start.min(chars.len())..end.min(chars.len())].iter().collect()
}
