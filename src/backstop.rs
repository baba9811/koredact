//! Opt-in regex safety net layered over the model (`backstop = true`). Deterministic and conservative:
//! it adds EMAIL / URL / partially-masked PHONE spans the model may have missed, plus a type-agnostic
//! `NUM` catch-all for long digit runs, and it protects template variables (`#{…}`, `{{…}}`, `${…}`)
//! so no span can cover them. Scores sit below any confident model span, so model spans win overlaps.
//! Offsets are char indices, like every other span in this crate.
use std::sync::LazyLock;

use regex::Regex;

use crate::types::{EntityType, Span};

pub const SCORE_TYPED: f32 = 0.6;   // EMAIL / URL / PHONE — below any confident model span
pub const SCORE_NUM: f32 = 0.5;     // NUM loses to every typed span it overlaps
pub const NUM_MIN_DIGITS: usize = 7; // shorter runs ("총 5분", "1:1", 5-digit postal codes) never fire

static VAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"#\{[^}]*\}|\{\{[^}]*\}\}|\$\{[^}]*\}").unwrap());
// ASCII local part and domain only — `\w` would swallow a following Korean particle ("user@x.test을").
static EMAIL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+").unwrap());
// Stops at whitespace, Hangul (a path in Hangul and a following particle are indistinguishable by
// character class, so Hangul is treated as prose), angle brackets, quotes and common bullet symbols.
static URL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(?:https?://|www\.)[^\s\x{3131}-\x{318e}\x{ac00}-\x{d7a3}<>"\x{201c}\x{201d}\x{2018}\x{2019}▶☎]+"#).unwrap()
});
const URL_TRAIL: &str = ".,;:!?…\u{3002}'";
// Partially masked mobile number, `010-****-1234` / `010-1234-****`. Only the 010 prefix (011/016-019
// collide with account and code prefixes) and exactly four stars; boundary digits are rejected below.
static PHONE_PARTIAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"010[-. ]?(?:\*{4}[-. ]?[0-9]{4}|[0-9]{4}[-. ]?\*{4})").unwrap());
// Digit run joined by at most one separator char per gap: "010 1234 5678" merges, "90. 85. 78" does not.
static RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[0-9]+(?:[-. ][0-9]+)*").unwrap());

/// Byte offsets of each char plus the text itself, so regex byte matches map back to char indices.
struct Text { s: String, byte_of_char: Vec<usize> }

impl Text {
    fn new(chars: &[char]) -> Text {
        let s: String = chars.iter().collect();
        let mut byte_of_char: Vec<usize> = s.char_indices().map(|(b, _)| b).collect();
        byte_of_char.push(s.len());
        Text { s, byte_of_char }
    }
    fn char_at(&self, byte: usize) -> usize {
        self.byte_of_char.binary_search(&byte).expect("regex match boundaries fall on char boundaries")
    }
}

/// Template-variable zones (char ranges) that must never be masked.
pub fn var_zones(chars: &[char]) -> Vec<(usize, usize)> {
    let t = Text::new(chars);
    VAR.find_iter(&t.s).map(|m| (t.char_at(m.start()), t.char_at(m.end()))).collect()
}

/// Trailing prose punctuation / quotes / unmatched closers are not part of a URL.
fn trim_url_end(seg: &str) -> usize {
    let chars: Vec<char> = seg.chars().collect();
    let mut end = chars.len();
    while end > 0 {
        let ch = chars[end - 1];
        if URL_TRAIL.contains(ch) { end -= 1; continue; }
        let opener = match ch { ')' => '(', ']' => '[', '}' => '{', _ => '\0' };
        if opener != '\0' {
            let head = &chars[..end];
            let opens = head.iter().filter(|c| **c == opener).count();
            let closes = head.iter().filter(|c| **c == ch).count();
            if opens < closes { end -= 1; continue; }
        }
        break;
    }
    end
}

/// Regex spans, sorted by (start, end). Overlaps between them are left for `mask::combine_backstop`.
pub fn find(chars: &[char]) -> Vec<Span> {
    let t = Text::new(chars);
    let mut out: Vec<Span> = Vec::new();
    for m in EMAIL.find_iter(&t.s) {
        out.push(Span::new(t.char_at(m.start()), t.char_at(m.end()), EntityType::Email, SCORE_TYPED));
    }
    for m in URL.find_iter(&t.s) {
        let start = t.char_at(m.start());
        let end = start + trim_url_end(m.as_str());
        if end > start { out.push(Span::new(start, end, EntityType::Url, SCORE_TYPED)); }
    }
    for m in PHONE_PARTIAL.find_iter(&t.s) {
        let (start, end) = (t.char_at(m.start()), t.char_at(m.end()));
        let digit_before = start > 0 && chars[start - 1].is_ascii_digit();
        let digit_after = end < chars.len() && chars[end].is_ascii_digit();
        if !digit_before && !digit_after { out.push(Span::new(start, end, EntityType::Phone, SCORE_TYPED)); }
    }
    for m in RUN.find_iter(&t.s) {
        if m.as_str().chars().filter(|c| c.is_ascii_digit()).count() >= NUM_MIN_DIGITS {
            out.push(Span::new(t.char_at(m.start()), t.char_at(m.end()), EntityType::Num, SCORE_NUM));
        }
    }
    out.sort_by_key(|s| (s.start, s.end));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::char_slice;

    fn chars(s: &str) -> Vec<char> { s.chars().collect() }
    fn typed(text: &str, t: EntityType) -> Vec<String> {
        let c = chars(text);
        find(&c).into_iter().filter(|s| s.entity == t).map(|s| char_slice(&c, s.start, s.end)).collect()
    }

    #[test]
    fn var_three_forms() {
        assert_eq!(var_zones(&chars("안녕 #{이름} 님 {{code}} ${url}")).len(), 3);
    }

    #[test]
    fn digit_run_email_url() {
        let mut ents: Vec<&str> = find(&chars("01012345678 a@b.com www.x.co/kr")).iter().map(|s| s.entity.as_str()).collect();
        ents.sort();
        assert_eq!(ents, ["EMAIL", "NUM", "URL"]);
    }

    #[test]
    fn short_runs_and_separated_numbers_do_not_fire() {
        for t in ["총 5분이면 끝나요", "1:1 상담", "카드로 3개월 무이자", "90. 85. 78. 65"] {
            assert!(find(&chars(t)).is_empty(), "{t}");
        }
    }

    #[test]
    fn phone_partial_shapes() {
        assert_eq!(typed("연락처 010-****-1234 로 문의", EntityType::Phone), ["010-****-1234"]);
        assert_eq!(typed("본인확인 010-1234-**** 완료", EntityType::Phone), ["010-1234-****"]);
        assert_eq!(typed("010****1234", EntityType::Phone), ["010****1234"]);
        assert_eq!(typed("010 **** 1234", EntityType::Phone), ["010 **** 1234"]);
        assert_eq!(typed("연락처 010-****-5678", EntityType::Phone), ["010-****-5678"]);
    }

    #[test]
    fn phone_partial_requires_010_stars_and_clean_boundaries() {
        for t in ["문의 01012345678", "02-****-1234", "환불계좌 농협 011-1234-**** 로 입금",
                  "발급코드 016-2210-**** 확인", "계좌번호 010-1234-****56 입니다"] {
            assert!(typed(t, EntityType::Phone).is_empty(), "{t}");
        }
    }

    #[test]
    fn email_and_url_stop_before_particles_and_trailing_punctuation() {
        let cases = [
            ("메일 user@example.test을 확인", EntityType::Email, "user@example.test"),
            ("메일 first.last+tag@sub.example.co.kr로 전달", EntityType::Email, "first.last+tag@sub.example.co.kr"),
            ("링크 https://example.test/a/1234)을 눌러", EntityType::Url, "https://example.test/a/1234"),
            ("https://example.test/a, 문의", EntityType::Url, "https://example.test/a"),
            ("안내 www.example.test/path?x=1&y=2. 끝", EntityType::Url, "www.example.test/path?x=1&y=2"),
            ("[https://example.test/q] 참고", EntityType::Url, "https://example.test/q"),
            ("링크 https://example.test/a를▶ 확인", EntityType::Url, "https://example.test/a"),
        ];
        for (text, t, want) in cases {
            assert_eq!(typed(text, t), [want], "{text}");
        }
    }

    #[test]
    fn email_ascii_shapes() {
        for text in ["a@b.com", "a.b-c_d@e-f.gh.ij", "A1@B2.CO"] {
            assert_eq!(typed(text, EntityType::Email), [text]);
        }
    }

    #[test]
    fn url_balanced_brackets_ipv6_and_unmatched_closers() {
        let cases = [
            ("참고 https://example.test/a_(b) 링크", "https://example.test/a_(b)"),
            ("참고 https://example.test/a_(b)) 링크", "https://example.test/a_(b)"),
            ("(https://example.test/a) 링크", "https://example.test/a"),
            ("https://[::1]/health 확인", "https://[::1]/health"),
            ("https://example.test/it's-ok 확인", "https://example.test/it's-ok"),
            ("'https://example.test/q' 확인", "https://example.test/q"),
            ("https://example.test/x?q=1! 끝", "https://example.test/x?q=1"),
            ("https://example.test/path/ 끝", "https://example.test/path/"),
        ];
        for (text, want) in cases {
            assert_eq!(typed(text, EntityType::Url), [want], "{text}");
        }
    }

    #[test]
    fn hangul_is_a_prose_boundary_for_urls() {
        let text = "https://example.test/한글경로 확인, https://예시.한국 참고";
        assert_eq!(typed(text, EntityType::Url), ["https://example.test/"]);
    }
}
