//! Embedding shortlist for high-cardinality Choice (`afm-de-latest`).
//!
//! Matches `afm_d.de.shortlist`: top-k by state↔option cosine similarity, then one DecisionModel
//! forward on the narrowed option set. Confidence is discounted by `log(k)/log(K)`.

use crate::Error;

/// Default trigger and width from AFM-D Encoder.
pub const DEFAULT_THRESHOLD: usize = 40;
pub const DEFAULT_K: usize = 20;

/// Whether Choice with `n` options should shortlist.
pub fn needs_shortlist(n: usize, threshold: usize) -> bool {
    n >= threshold
}

/// Top-k option indices by cosine similarity of L2-normalized embedding rows.
///
/// `vectors` is `[state, opt0, opt1, …]` each of length `dim`. `keep` indices are forced in
/// (replacing the lowest-ranked slot), as in AFM-D labeled eval.
pub fn shortlist_indices(
    vectors: &[Vec<f32>],
    k: usize,
    keep: &[usize],
) -> Result<Vec<usize>, Error> {
    if vectors.is_empty() {
        return Err(Error::invalid("shortlist: empty vectors"));
    }
    let n_opts = vectors.len() - 1;
    if k < 1 {
        return Err(Error::invalid("shortlist k must be >= 1"));
    }
    if k >= n_opts {
        return Ok((0..n_opts).collect());
    }
    let state = &vectors[0];
    let mut scores: Vec<(usize, f32)> = (0..n_opts)
        .map(|i| (i, dot(state, &vectors[i + 1])))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut order: Vec<usize> = scores.into_iter().take(k).map(|(i, _)| i).collect();
    for &idx in keep {
        if idx >= n_opts || order.contains(&idx) {
            continue;
        }
        if let Some(last) = order.last_mut() {
            *last = idx;
        }
    }
    let mut seen = std::collections::HashSet::new();
    order.retain(|&i| seen.insert(i));
    Ok(order)
}

/// AFM-D shortlist confidence discount: `log(k)/log(K)`.
pub fn confidence_discount(k: usize, capital_k: usize) -> f64 {
    if capital_k <= 1 || k == 0 {
        return 1.0;
    }
    (k as f64).ln() / (capital_k as f64).ln()
}

/// L2-normalize each row in place.
pub fn l2_normalize(rows: &mut [Vec<f32>]) {
    for row in rows {
        let n: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in row.iter_mut() {
                *x /= n;
            }
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Deterministic bag-of-token-id embeddings for tests / convert without an embed graph.
/// Each text → unit vector over a hashed bag (dim 256).
pub fn bag_embed(token_ids: &[u32], dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    for &id in token_ids {
        let i = (id as usize) % dim;
        v[i] += 1.0;
    }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_closest() {
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.8, 0.2],
        ];
        let idx = shortlist_indices(&vectors, 2, &[]).unwrap();
        assert_eq!(idx, vec![0, 2]);
    }

    #[test]
    fn discount() {
        let d = confidence_discount(20, 255);
        assert!((d - (20f64.ln() / 255f64.ln())).abs() < 1e-9);
    }
}
