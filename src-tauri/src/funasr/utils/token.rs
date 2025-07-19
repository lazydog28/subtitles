use anyhow::Result;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
/// Token ID Converter
pub struct TokenIdConverter {
    token_list: Vec<String>,
    unk_symbol: String,
}

impl TokenIdConverter {
    pub fn new(token_list: Vec<String>) -> Self {
        let unk_symbol = token_list.last().unwrap().clone();
        Self {
            token_list,
            unk_symbol,
        }
    }

    pub fn ids2tokens(&self, integers: &[usize]) -> Vec<String> {
        integers
            .iter()
            .map(|&i| self.token_list.get(i).unwrap_or(&self.unk_symbol).clone())
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
