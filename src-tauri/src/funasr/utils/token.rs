use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
/// Token ID Converter
pub struct TokenIdConverter {
    token_list: Vec<String>,
    unk_symbol: String,
    token2id: HashMap<String, usize>,
    unk_id: usize,
}

impl TokenIdConverter {
    pub fn new(token_list: Vec<String>) -> Self {
        let unk_symbol = token_list.last().unwrap().clone();
        let token2id: HashMap<String, usize> = token_list
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();
        let unk_id = token2id[&unk_symbol];

        Self {
            token_list,
            unk_symbol,
            token2id,
            unk_id,
        }
    }

    pub fn get_num_vocabulary_size(&self) -> usize {
        self.token_list.len()
    }

    pub fn ids2tokens(&self, integers: &[usize]) -> Vec<String> {
        integers
            .iter()
            .map(|&i| self.token_list.get(i).unwrap_or(&self.unk_symbol).clone())
            .collect()
    }

    pub fn tokens2ids(&self, tokens: &[String]) -> Vec<usize> {
        tokens
            .iter()
            .map(|token| self.token2id.get(token).copied().unwrap_or(self.unk_id))
            .collect()
    }
}

pub fn read_token(path: impl AsRef<Path>) -> Result<TokenIdConverter> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut token_list = Vec::new();

    for line in reader.lines() {
        let line = line?;
        // 空格分隔 获取第一个
        let token = line.split_whitespace().next().unwrap_or("").trim();
        if token.is_empty() {
            continue; // 跳过空行
        }
        token_list.push(token.to_string());
    }

    Ok(TokenIdConverter::new(token_list))
}
