use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::MlResult;

const MAGIC: &[u8; 8] = b"TRNCHDP\0";
const FORMAT_VERSION: u32 = 1;

// ── 직렬화 타입 ─────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelState {
    pub version: String,
    pub layers: Vec<LayerState>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerState {
    pub layer_type: String,
    pub label: String,
    pub config: serde_json::Value,
    pub params: Vec<ParamState>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ParamState {
    pub name: String,
    pub shape: Vec<usize>,
    /// JSON 포맷에서 실제 가중치 데이터. binary 포맷에서는 비어있음.
    #[serde(default)]
    pub data: Vec<f32>,
    /// binary(.tdw) 포맷 전용: blob 내 f32 시작 인덱스
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob_offset: Option<u64>,
    /// binary(.tdw) 포맷 전용: blob 내 f32 개수
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob_length: Option<u64>,
}

// ── ModelState 저장/로드 ─────────────────────────────────────────────────────

impl ModelState {
    pub fn new(layers: Vec<LayerState>) -> Self {
        Self {
            version: FORMAT_VERSION.to_string(),
            layers,
        }
    }

    /// 확장자에 따라 JSON(.json) 또는 binary(.tdw)로 자동 저장
    pub fn save(&self, path: &str) -> MlResult<()> {
        match Path::new(path).extension().and_then(|e| e.to_str()) {
            Some("tdw") => self.save_binary(path),
            _           => self.save_json(path),
        }
    }

    /// 확장자에 따라 JSON(.json) 또는 binary(.tdw)로 자동 로드
    pub fn load(path: &str) -> MlResult<Self> {
        match Path::new(path).extension().and_then(|e| e.to_str()) {
            Some("tdw") => Self::load_binary(path),
            _           => Self::load_json(path),
        }
    }

    // ── JSON ────────────────────────────────────────────────────────────────

    fn save_json(&self, path: &str) -> MlResult<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::MlError::StringError(e.to_string()))?;
        ensure_parent_dir(path)?;
        std::fs::write(path, json)
            .map_err(|e| crate::MlError::StringError(
                format!("파일 쓰기 실패 '{}': {}", path, e)
            ))
    }

    fn load_json(path: &str) -> MlResult<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| crate::MlError::StringError(
                format!("파일 읽기 실패 '{}': {}", path, e)
            ))?;
        serde_json::from_str(&json)
            .map_err(|e| crate::MlError::StringError(
                format!("JSON 파싱 실패 '{}': {}", path, e)
            ))
    }

    // ── Binary (.tdw) ───────────────────────────────────────────────────────

    /// 포맷: [magic 8B][version u32 LE][header_len u32 LE][header JSON][weight blob]
    fn save_binary(&self, path: &str) -> MlResult<()> {
        use std::io::Write;

        // weight blob 빌드 + 헤더용 offset/length 채우기
        let mut blob: Vec<u8> = Vec::new();
        let mut header_layers: Vec<LayerState> = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let mut header_params: Vec<ParamState> = Vec::with_capacity(layer.params.len());
            for param in &layer.params {
                let offset = (blob.len() / 4) as u64;   // f32 단위 오프셋
                let length = param.data.len() as u64;
                for &v in &param.data {
                    blob.extend_from_slice(&v.to_le_bytes());
                }
                header_params.push(ParamState {
                    name: param.name.clone(),
                    shape: param.shape.clone(),
                    data: vec![],
                    blob_offset: Some(offset),
                    blob_length: Some(length),
                });
            }
            header_layers.push(LayerState {
                layer_type: layer.layer_type.clone(),
                label: layer.label.clone(),
                config: layer.config.clone(),
                params: header_params,
            });
        }

        let header_state = ModelState { version: self.version.clone(), layers: header_layers };
        let header_json = serde_json::to_vec(&header_state)
            .map_err(|e| crate::MlError::StringError(e.to_string()))?;

        ensure_parent_dir(path)?;
        let mut file = std::fs::File::create(path)
            .map_err(|e| crate::MlError::StringError(
                format!("파일 생성 실패 '{}': {}", path, e)
            ))?;

        file.write_all(MAGIC)
            .and_then(|_| file.write_all(&FORMAT_VERSION.to_le_bytes()))
            .and_then(|_| file.write_all(&(header_json.len() as u32).to_le_bytes()))
            .and_then(|_| file.write_all(&header_json))
            .and_then(|_| file.write_all(&blob))
            .map_err(|e| crate::MlError::StringError(
                format!("바이너리 쓰기 실패 '{}': {}", path, e)
            ))
    }

    fn load_binary(path: &str) -> MlResult<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| crate::MlError::StringError(
                format!("파일 읽기 실패 '{}': {}", path, e)
            ))?;

        // magic 검증
        if bytes.len() < 16 || &bytes[0..8] != MAGIC {
            return Err(crate::MlError::StringError(
                format!("'{}' 은 올바른 .tdw 파일이 아님 (magic 불일치)", path)
            ));
        }

        // 버전 검증
        let file_version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if file_version != FORMAT_VERSION {
            return Err(crate::MlError::StringError(format!(
                "지원하지 않는 .tdw 버전: {} (현재 지원: {})", file_version, FORMAT_VERSION
            )));
        }

        let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let blob_start = 16 + header_len;
        if bytes.len() < blob_start {
            return Err(crate::MlError::StringError(
                format!("'{}' 파일이 잘렸거나 손상됨", path)
            ));
        }

        let mut state: ModelState = serde_json::from_slice(&bytes[16..blob_start])
            .map_err(|e| crate::MlError::StringError(
                format!("헤더 파싱 실패 '{}': {}", path, e)
            ))?;

        // blob에서 실제 데이터 복원
        let blob = &bytes[blob_start..];
        for layer in &mut state.layers {
            for param in &mut layer.params {
                if let (Some(offset), Some(length)) =
                    (param.blob_offset.take(), param.blob_length.take())
                {
                    let byte_start = offset as usize * 4;
                    let byte_end   = byte_start + length as usize * 4;
                    if byte_end > blob.len() {
                        return Err(crate::MlError::StringError(format!(
                            "파라미터 '{}' blob 오프셋이 파일 범위를 벗어남", param.name
                        )));
                    }
                    param.data = blob[byte_start..byte_end]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                }
            }
        }

        Ok(state)
    }
}

// ── 헬퍼 ────────────────────────────────────────────────────────────────────

fn ensure_parent_dir(path: &str) -> MlResult<()> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| crate::MlError::StringError(
                    format!("디렉토리 생성 실패 '{}': {}", parent.display(), e)
                ))?;
        }
    }
    Ok(())
}

/// params 슬라이스에서 이름으로 ParamState를 찾음
pub(crate) fn find_param<'a>(params: &'a [ParamState], name: &str) -> MlResult<&'a ParamState> {
    params.iter().find(|p| p.name == name)
        .ok_or_else(|| crate::MlError::StringError(
            format!("파라미터 '{}' 를 체크포인트에서 찾지 못함", name)
        ))
}

/// 저장된 shape와 현재 모델의 shape를 비교
pub(crate) fn validate_shape(param: &ParamState, expected: &[usize]) -> MlResult<()> {
    if param.shape.as_slice() != expected {
        Err(crate::MlError::StringError(format!(
            "파라미터 '{}' shape 불일치: 파일={:?}, 현재 모델={:?}",
            param.name, param.shape, expected
        )))
    } else {
        Ok(())
    }
}
